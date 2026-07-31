"""Wrapper to run guardian orchestrator with environment variables."""
import os
import sys
import subprocess

def main():
    # Set environment variables from command line args
    if len(sys.argv) > 1 and sys.argv[1].startswith('TWITTER_AUTH_COOKIE='):
        os.environ['TWITTER_AUTH_COOKIE'] = sys.argv[1].split('=', 1)[1]
    
    if len(sys.argv) > 2 and sys.argv[2].startswith('LANGSMITH_API_KEY='):
        os.environ['LANGSMITH_API_KEY'] = sys.argv[2].split('=', 1)[1]
    
    # Get the remaining args (the actual Python command)
    python_args = sys.argv[3:]
    
    # Run the orchestrator
    subprocess.run([sys.executable] + python_args)

if __name__ == '__main__':
    main()
