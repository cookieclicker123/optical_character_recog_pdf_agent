#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys
import shutil
import logging
import tempfile

logger = logging.getLogger(__name__)

class OCRRunner:
    def __init__(self):
        self.base_dir = Path.cwd()
        # Create persistent tmp directory for venvs
        self.tmp_dir = self.base_dir / 'tmp'
        self.tmp_dir.mkdir(exist_ok=True)
        
        self.venvs = {
            'tesseract': {
                'name': '.venv_tesseract',
                'requirements': 'requirements_tesseract.txt',
            },
            'mllm': {
                'name': '.venv_mllm',
                'requirements': 'requirements_mllm.txt',
            }
        }

    def get_venv_path(self, venv_type: str) -> Path:
        """Get path to virtual environment in tmp directory"""
        return self.tmp_dir / self.venvs[venv_type]['name']

    def setup_venv(self, venv_type: str) -> bool:
        """Create and setup virtual environment in tmp directory"""
        try:
            venv_config = self.venvs[venv_type]
            venv_path = self.get_venv_path(venv_type)
            req_path = self.base_dir / venv_config['requirements']

            # Remove existing venv if it exists
            if venv_path.exists():
                shutil.rmtree(venv_path)

            # Create venv
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)

            # Install requirements
            pip_path = venv_path / ('Scripts' if sys.platform == 'win32' else 'bin') / 'pip'
            subprocess.run([str(pip_path), 'install', '-U', 'pip'], check=True)
            subprocess.run([str(pip_path), 'install', '-r', str(req_path)], check=True)
            
            print(f"✅ Created {venv_type} environment in {venv_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to create {venv_type} environment: {e}")
            return False

    def run_pipeline(self, venv_type: str, input_dir: Path, output_dir: Path, mode: str = 'test', compare: bool = False):
        """Run the OCR pipeline in the appropriate environment with specified mode"""
        venv_path = self.get_venv_path(venv_type)
        python_path = venv_path / ('Scripts' if sys.platform == 'win32' else 'bin') / 'python'
        pytest_path = venv_path / ('Scripts' if sys.platform == 'win32' else 'bin') / 'pytest'

        if not python_path.exists():
            print(f"❌ Environment not found. Creating {venv_type} environment first...")
            if not self.setup_venv(venv_type):
                return

        try:
            if venv_type == 'mllm':
                print(f"\n🚀 Running MLLM {mode}...")
                
                if mode == 'test':
                    # Run MLLM test files
                    test_files = [
                        'tests/test_data_model.py',
                        'tests/test_mock_OCR.py',
                        'tests/test_vision_tool.py',
                        'tests/test_OCR.py',
                        'tests/test_translate.py',
                        'tests/test_json_grouping.py'
                    ]
                    
                    for test_file in test_files:
                        test_path = self.base_dir / test_file
                        if test_path.exists():
                            print(f"\nRunning {test_file}...")
                            subprocess.run([
                                str(pytest_path),
                                str(test_path),
                                '-v'  # verbose output
                            ], check=True)
                        else:
                            print(f"⚠️ Test file not found: {test_file}")
                
                elif mode in ['cli', 'api']:
                    print(f"ℹ️ MLLM {mode} implementation not yet available")
                
            elif venv_type == 'tesseract':
                # Change to the tesseract implementation directory
                implementation_dir = self.base_dir / 'examples' / 'PyTesseract_Google_Translate_implementation'
                
                if not implementation_dir.exists():
                    raise FileNotFoundError(f"Tesseract implementation not found at {implementation_dir}")
                
                # Run each test file
                test_files = [
                    'tests/test_mock_OCR.py',
                    'tests/test_OCR.py',
                    'tests/test_mock_llm_information_grouping.py',
                    'tests/test_llm_information_grouping.py',
                    'tests/test_groq_basic.py'
                ]
                
                for test_file in test_files:
                    test_path = implementation_dir / test_file
                    if test_path.exists():
                        print(f"\nRunning {test_file}...")
                        subprocess.run([
                            str(pytest_path),
                            str(test_path),
                            '-v'  # verbose output
                        ], cwd=str(implementation_dir), check=True)
                    else:
                        print(f"⚠️ Test file not found: {test_file}")
                
            print(f"✅ {venv_type} {mode} completed")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Test execution failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    def run_comparison(self, input_dir: Path, output_dir: Path):
        """Run both pipelines and organize outputs for comparison"""
        print("\n📊 Running comparison of both pipelines...")
        self.run_pipeline('tesseract', input_dir, output_dir, compare=True)
        self.run_pipeline('mllm', input_dir, output_dir, compare=True)
        print("\n✅ Comparison complete. Results in separate directories under output.")

    def cleanup_venvs(self):
        """Clean up all virtual environments in tmp directory"""
        try:
            if self.tmp_dir.exists():
                shutil.rmtree(self.tmp_dir)
                self.tmp_dir.mkdir(exist_ok=True)
                print("✅ Cleaned up all virtual environments")
                return True
        except Exception as e:
            print(f"❌ Failed to cleanup environments: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='OCR Pipeline Runner and Environment Manager')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup virtual environments')
    setup_parser.add_argument('venv_type', choices=['tesseract', 'mllm', 'all'])

    # Run command
    run_parser = subparsers.add_parser('run', help='Run OCR pipeline')
    run_parser.add_argument('implementation', 
                          choices=['tesseract', 'mllm', 'compare'],
                          help='Choose implementation or compare both')
    run_parser.add_argument('--mode',
                          choices=['test', 'cli', 'api'],
                          default='test',
                          help='Mode to run (test, cli, or api)')
    run_parser.add_argument('--input-dir', type=Path, required=True)
    run_parser.add_argument('--output-dir', type=Path, required=True)

    # Cleanup command
    subparsers.add_parser('cleanup', help='Remove all virtual environments')

    args = parser.parse_args()
    runner = OCRRunner()

    if args.command == 'setup':
        if args.venv_type == 'all':
            for venv in ['tesseract', 'mllm']:
                runner.setup_venv(venv)
        else:
            runner.setup_venv(args.venv_type)

    elif args.command == 'run':
        if args.implementation == 'compare':
            runner.run_comparison(args.input_dir, args.output_dir)
        else:
            runner.run_pipeline(args.implementation, args.input_dir, args.output_dir, mode=args.mode)

    elif args.command == 'cleanup':
        runner.cleanup_venvs()

if __name__ == '__main__':
    main()
