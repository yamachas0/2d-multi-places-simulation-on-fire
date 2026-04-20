"""
LLM-based agent in 2D worlds with multiple places.
"""
import argparse
import datetime
import logging
import re
import yaml
import os
import shutil
import time
import numpy as np
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from simulation import Simulation
from visualization import Visualizer
from reporter import build_report

# Constants
DEFAULT_FRAME_INTERVAL_INTERACTIVE = 10
DEFAULT_FRAME_INTERVAL_CONFIG = 50
VISUALIZATION_UPDATE_DELAY = 0.2
SIMULATIONS_ROOT = "simulations"


def _next_run_sequence(simulations_root: str) -> int:
    """Scan simulations/ for existing `{date}_{NN}_...` dirs and return the next NN."""
    if not os.path.isdir(simulations_root):
        return 1
    max_seq = 0
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}_(\d{2,})_")
    for name in os.listdir(simulations_root):
        m = pat.match(name)
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    return max_seq + 1


def _derive_run_name(config_path: str, config: dict) -> str:
    """Pick a short run name. Prefer config.visualization.run_name, else strip
    the config filename of leading 'config_' and the extension."""
    vis = config.get('visualization', {}) or {}
    if vis.get('run_name'):
        return str(vis['run_name'])
    stem = os.path.splitext(os.path.basename(config_path))[0]
    if stem.startswith('config_'):
        stem = stem[len('config_'):]
    return stem or 'run'


def resolve_run_dir(config_path: str, config: dict) -> Tuple[str, str]:
    """Return (output_dir_path, basename). output_dir is the run folder,
    basename is the folder name (used to rename html/md/transcript)."""
    vis = config.get('visualization', {}) or {}
    # Backwards-compat: configs that explicitly set visualization.output_dir
    # keep the old flat layout (used by dev / smoke tests).
    if vis.get('output_dir'):
        output_dir = vis['output_dir']
        return output_dir, os.path.basename(os.path.normpath(output_dir))

    dt_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    seq = _next_run_sequence(SIMULATIONS_ROOT)
    name = _derive_run_name(config_path, config)
    folder = f"{dt_str}_{seq:02d}_{name}"
    output_dir = os.path.join(SIMULATIONS_ROOT, folder)
    return output_dir, folder


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if 'log_file' in log_config:
        handlers.append(logging.FileHandler(log_config['log_file']))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )


def check_llm_setup(sim: Simulation, logger: logging.Logger) -> bool:
    """Check LLM API connection and model availability"""
    if not sim.llm_client.check_connection():
        logger.error("Cannot connect to LLM API. Check ANTHROPIC_API_KEY and network.")
        logger.error(f"Expected URL: {sim.llm_client.base_url}")
        return False

    if not sim.llm_client.check_model_exists():
        logger.warning(f"Model '{sim.llm_client.model}' not available.")
        return False

    logger.info(f"Using model: {sim.llm_client.model}")
    return True


def determine_visualization_settings(args, config: dict, config_path: str) -> Tuple[bool, bool, int, str, str]:
    """Determine visualization settings from args and config.
    Returns (should_visualize, config_save_frames, frame_interval, output_dir, basename)."""
    config_save_frames = config.get('visualization', {}).get('save_frames', False)
    should_visualize = args.visualize or args.save_frames or config_save_frames

    frame_interval = (
        args.frame_interval or
        config.get('visualization', {}).get('frame_interval', DEFAULT_FRAME_INTERVAL_CONFIG)
    )

    # For interactive visualization, use smaller interval
    if args.visualize and not args.save_frames and not args.frame_interval:
        frame_interval = DEFAULT_FRAME_INTERVAL_INTERACTIVE

    output_dir, basename = resolve_run_dir(config_path, config)

    return should_visualize, config_save_frames, frame_interval, output_dir, basename


def handle_visualization(
    visualizer: Visualizer,
    sim: Simulation,
    step: int,
    frame_interval: int,
    should_save: bool,
    output_dir: str,
    logger: logging.Logger
):
    """Handle visualization for a simulation step"""
    if step % frame_interval != 0 and step != sim.duration - 1:
        return
    
    place_status = sim.get_place_status()
    
    time_str = sim._current_time_str() if hasattr(sim, '_current_time_str') else None
    step_messages = getattr(sim, 'last_step_messages', None)

    if should_save:
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        save_path = os.path.join(frames_dir, f"frame_{step:04d}.png")
        visualizer.visualize_step(
            sim.agents,
            place_status,
            step,
            communication_radius=sim.communication_radius,
            save_path=save_path,
            fire_states=sim.fire_states,
            time_str=time_str,
            step_messages=step_messages,
        )
        logger.info(f"Saved frame: {save_path}")
    else:
        # Interactive visualization only
        logger.info(f"Displaying visualization for step {step}")
        try:
            visualizer.visualize_step(
                sim.agents,
                place_status,
                step,
                communication_radius=sim.communication_radius,
                fire_states=sim.fire_states,
                time_str=time_str,
                step_messages=step_messages,
            )
            time.sleep(VISUALIZATION_UPDATE_DELAY)
        except Exception as e:
            logger.error(f"Error displaying visualization: {e}", exc_info=True)


def print_statistics(stats: dict, sim: Simulation, logger: logging.Logger):
    """Print simulation statistics"""
    logger.info("\n=== Simulation Statistics ===")
    logger.info(f"Total steps: {stats.get('total_steps', 0)}")
    logger.info(f"Overall mean occupancy: {stats.get('mean_occupancy', 0):.2%}")
    logger.info(f"Overall std occupancy: {stats.get('std_occupancy', 0):.2%}")
    logger.info(f"Mean agents in places: {stats.get('mean_agents_in_place', 0):.2f}")
    logger.info(f"Max agents in places: {stats.get('max_agents_in_place', 0)}")
    logger.info(f"Min agents in places: {stats.get('min_agents_in_place', 0)}")
    
    # Print per-place statistics
    if 'places' in sim.stats:
        logger.info("\n=== Per-Place Statistics ===")
        for place_name, place_stats in sim.stats['places'].items():
            if place_stats['occupancy']:
                occupancy_array = np.array(place_stats['occupancy'])
                agents_array = np.array(place_stats['agents_in_place'])
                logger.info(f"\n{place_name}:")
                logger.info(f"  Mean occupancy: {np.mean(occupancy_array):.2%}")
                logger.info(f"  Mean agents: {np.mean(agents_array):.2f}")
                logger.info(f"  Max agents: {int(np.max(agents_array))}")
                logger.info(f"  Min agents: {int(np.min(agents_array))}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simulation of LLM-based agent in 2D worlds with multiple places.')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization during simulation'
    )
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save visualization frames'
    )
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=None,
        help='Interval between visualization frames (overrides config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (overrides simulation.seed in config)'
    )

    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Determine visualization settings
    should_visualize, config_save_frames, frame_interval, output_dir, run_basename = \
        determine_visualization_settings(args, config, args.config)
    
    # Remove output directory if it exists
    if os.path.exists(output_dir):
        logger.info(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Create output directory if needed
    if args.save_frames or config_save_frames:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        # Freeze the config used for this run alongside its outputs.
        try:
            shutil.copy2(args.config, os.path.join(output_dir, "config.yaml"))
        except Exception as e:
            logger.warning(f"Could not copy config into run dir: {e}")
    
    # Initialize simulation
    sim = Simulation(config_path=args.config, output_dir=output_dir, seed=args.seed)
    
    # Initialize visualizer if needed
    visualizer = None
    if should_visualize:
        focus_agent_id = config.get('visualization', {}).get('focus_agent_id')
        visualizer = Visualizer(
            half_space_size=sim.half_space_size,
            places=sim.places,
            num_agents=sim.num_agents,
            focus_agent_id=focus_agent_id,
        )
    
    # Run simulation
    try:
        # Initialize agents
        sim.initialize_agents()
        
        # Check LLM setup
        if not check_llm_setup(sim, logger):
            return
        
        logger.info("Starting simulation...")
        
        # Run simulation steps
        while sim.step < sim.duration:
            sim.step_simulation()
            
            # Visualize if needed
            if visualizer and should_visualize:
                should_save = args.save_frames or (config_save_frames and not args.visualize)
                handle_visualization(
                    visualizer, sim, sim.step, frame_interval,
                    should_save, output_dir, logger
                )
        
        logger.info("Simulation completed")

        # Feature 5: export unified JSON dataset for the Canvas viewer.
        try:
            exported = sim.export_simulation_data()
            if exported:
                logger.info(f"Viewer data: {exported}")
                # Bundle viewer_v2.html + inlined JSON into a single
                # file so it runs from file:// without a local server.
                try:
                    from tools.bundle_viewer import bundle_viewer
                    bundled = bundle_viewer(output_dir)
                    logger.info(f"Bundled viewer: {bundled}")
                except Exception as e:
                    logger.warning(f"Viewer bundle skipped: {e}")
        except Exception as e:
            logger.error(f"Failed to export simulation_data.json: {e}", exc_info=True)

        # Print statistics
        stats = sim.get_statistics()
        print_statistics(stats, sim, logger)
        
        # Plot statistics
        if visualizer:
            should_save_stats = args.save_frames or config_save_frames
            stats_path = None
            if should_save_stats:
                frames_dir = os.path.join(output_dir, 'frames')
                os.makedirs(frames_dir, exist_ok=True)
                stats_path = os.path.join(frames_dir, 'statistics.png')
            visualizer.plot_statistics(sim.stats, save_path=stats_path, fire_states=sim.fire_states)
            if stats_path:
                logger.info(f"Saved statistics plot: {stats_path}")

        # Build single-page HTML report (GIF + conversation + thinking)
        if args.save_frames or config_save_frames:
            try:
                report_path = build_report(
                    output_dir, config, total_steps=sim.step, basename=run_basename
                )
                logger.info(f"Open in browser: {report_path}")
            except Exception as e:
                logger.error(f"Failed to build report: {e}", exc_info=True)
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)


if __name__ == "__main__":
    main()

