#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path
from Acquisition_function import optimizer

def main():
    """Main optimization runner"""
    print("=== Starting Bayesian Optimization ===")
    
    # Create results directory
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Change to UCB directory if it exists
    ucb_dir = Path("UCB")
    if ucb_dir.exists():
        print(f"Changing to UCB directory: {ucb_dir}")
        os.chdir(ucb_dir)
    else:
        print("UCB directory not found, running in current directory")
    
    # Display optimization settings
    print("\nOptimization Settings:")
    print(f"  Parameter bounds: {optimizer.pbounds}")
    print(f"  Acquisition function: {type(optimizer.acquisition_function).__name__}")
    print(f"  Random state: {optimizer.random_state}")
    
    # Run optimization
    print("\n=== Running Bayesian Optimization ===")
    try:
        optimizer.maximize(init_points=5, n_iter=15)
        print("Optimization completed successfully!")
    except Exception as e:
        print(f"Error during optimization: {e}")
        return 1
    
    # Get best results
    best_params = optimizer.max['params']
    best_score = optimizer.max['target']
    
    print(f"\n=== Optimization Results ===")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.3f}")
    
    # Prepare detailed results
    results = {
        "best_params": best_params,
        "best_score": best_score,
        "optimization_settings": {
            "init_points": 5,
            "n_iter": 15,
            "parameter_bounds": optimizer.pbounds,
            "acquisition_function": type(optimizer.acquisition_function).__name__,
            "random_state": optimizer.random_state
        },
        "all_results": [
            {
                "iteration": i,
                "params": point["params"], 
                "target": point["target"]
            } 
            for i, point in enumerate(optimizer.res, 1)
        ]
    }
    
    # Save results
    results_file = results_dir / "optimization_results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
        return 1
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    scores = [point["target"] for point in optimizer.res]
    print(f"Total iterations: {len(scores)}")
    print(f"Best score: {max(scores):.3f}")
    print(f"Average score: {sum(scores)/len(scores):.3f}")
    print(f"Worst score: {min(scores):.3f}")
    
    # Show top 3 results
    print(f"\n=== Top 3 Results ===")
    sorted_results = sorted(results["all_results"], key=lambda x: x["target"], reverse=True)
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Score: {result['target']:.3f}, Params: {result['params']}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)