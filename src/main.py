#!/usr/bin/env python3
"""
ImgGen AI - Main Application Entry Point
High-fidelity image generation and editing system
"""

import argparse
import sys
from pathlib import Path

from core.app import ImgGenApp
from ui.interface import launch_interface
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="ImgGen AI - Image Generation System")
    parser.add_argument("--mode", choices=["ui", "api", "cli"], default="ui",
                       help="Launch mode: ui (ComfyUI), api (REST API), or cli (command line)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8188, help="Port number")
    parser.add_argument("--config", default="config/default.yaml", help="Configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(level=log_level)
    
    try:
        # Initialize the application
        app = ImgGenApp(config_path=args.config)
        
        if args.mode == "ui":
            logger.info("Launching ComfyUI interface...")
            launch_interface(host=args.host, port=args.port)
        elif args.mode == "api":
            logger.info("Starting API server...")
            app.start_api_server(host=args.host, port=args.port)
        elif args.mode == "cli":
            logger.info("Starting CLI mode...")
            app.start_cli()
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()