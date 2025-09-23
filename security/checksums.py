#!/usr/bin/env python3
"""
Comprehensive Checksum Generation and Verification System
Iusmorfos Framework - Security and Integrity Module

Author: Iusmorfos Development Team
Version: 1.0.0
"""

import hashlib
import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security/integrity.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ChecksumGenerator:
    """Multi-algorithm checksum generator for file integrity verification."""
    
    ALGORITHMS = {
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512,
        'blake2b': hashlib.blake2b
    }
    
    CRITICAL_FILES = [
        'src/*.py',
        'app/*.py', 
        'tests/*.py',
        'scripts/*.py',
        'config/*.yaml',
        'config/*.json',
        'Dockerfile',
        'requirements.txt',
        'README.md',
        'REPRODUCIBILITY.md',
        'CONTRIBUTING.md',
        'SECURITY.md',
        'data/**/*.csv',
        'data/**/*.json',
        'ro-crate-metadata.json',
        'notebooks/*.ipynb',
        'security/*.py',
        'security/*.json',
    ]
    
    def __init__(self, base_path: Union[str, Path] = '.'):
        """Initialize checksum generator."""
        self.base_path = Path(base_path).resolve()
        self.checksums_file = self.base_path / 'security' / 'checksums.json'
        self.integrity_report = self.base_path / 'security' / 'integrity_report.json'
        
        # Ensure security directory exists
        self.checksums_file.parent.mkdir(exist_ok=True)
        
        logger.info(f"Initialized ChecksumGenerator for: {self.base_path}")
    
    def calculate_file_checksums(self, file_path: Path) -> Dict[str, str]:
        """Calculate checksums for a single file."""
        checksums = {}
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                
            for alg_name, alg_func in self.ALGORITHMS.items():
                hash_obj = alg_func()
                hash_obj.update(content)
                checksums[alg_name] = hash_obj.hexdigest()
                
        except Exception as e:
            logger.error(f"Error calculating checksums for {file_path}: {e}")
            return {}
            
        return checksums
    
    def get_critical_files(self) -> List[Path]:
        """Get list of critical files for checksum verification."""
        critical_files = []
        
        for pattern in self.CRITICAL_FILES:
            abs_pattern = str(self.base_path / pattern)
            matches = glob.glob(abs_pattern, recursive=True)
            
            for match in matches:
                file_path = Path(match)
                if file_path.is_file() and file_path.exists():
                    critical_files.append(file_path)
        
        # Remove duplicates and sort
        critical_files = sorted(list(set(critical_files)))
        
        logger.info(f"Found {len(critical_files)} critical files for checksum verification")
        return critical_files
    
    def generate_all_checksums(self) -> Dict[str, Dict[str, str]]:
        """Generate checksums for all critical files."""
        all_checksums = {}
        critical_files = self.get_critical_files()
        
        logger.info(f"Generating checksums for {len(critical_files)} files...")
        
        for file_path in critical_files:
            relative_path = file_path.relative_to(self.base_path)
            checksums = self.calculate_file_checksums(file_path)
            
            if checksums:
                all_checksums[str(relative_path)] = {
                    'checksums': checksums,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'generated': datetime.now().isoformat()
                }
                logger.debug(f"Generated checksums for: {relative_path}")
            else:
                logger.warning(f"Failed to generate checksums for: {relative_path}")
        
        logger.info(f"✅ Generated checksums for {len(all_checksums)} files")
        return all_checksums
    
    def save_checksums(self, checksums: Dict[str, Dict[str, str]]) -> None:
        """Save checksums to file."""
        checksum_data = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'algorithms': list(self.ALGORITHMS.keys()),
                'repository': str(self.base_path),
                'file_count': len(checksums)
            },
            'files': checksums
        }
        
        try:
            with open(self.checksums_file, 'w') as f:
                json.dump(checksum_data, f, indent=2, sort_keys=True)
            
            logger.info(f"✅ Saved checksums to: {self.checksums_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving checksums: {e}")
            raise
    
    def load_checksums(self) -> Optional[Dict[str, Dict[str, str]]]:
        """Load existing checksums from file."""
        if not self.checksums_file.exists():
            logger.warning(f"Checksums file not found: {self.checksums_file}")
            return None
        
        try:
            with open(self.checksums_file, 'r') as f:
                data = json.load(f)
                
            logger.info(f"Loaded checksums for {len(data.get('files', {}))} files")
            return data.get('files', {})
            
        except Exception as e:
            logger.error(f"Error loading checksums: {e}")
            return None
    
    def verify_file_integrity(self, file_path: Path, stored_checksums: Dict[str, str]) -> Dict[str, bool]:
        """Verify integrity of a single file."""
        if not file_path.exists():
            logger.error(f"File not found for verification: {file_path}")
            return {alg: False for alg in self.ALGORITHMS.keys()}
        
        current_checksums = self.calculate_file_checksums(file_path)
        verification_results = {}
        
        for alg_name in self.ALGORITHMS.keys():
            stored_checksum = stored_checksums.get('checksums', {}).get(alg_name)
            current_checksum = current_checksums.get(alg_name)
            
            if stored_checksum and current_checksum:
                verification_results[alg_name] = (stored_checksum == current_checksum)
            else:
                verification_results[alg_name] = False
        
        return verification_results
    
    def verify_all_files(self) -> Dict[str, Dict[str, Union[bool, str]]]:
        """Verify integrity of all critical files."""
        stored_checksums = self.load_checksums()
        if not stored_checksums:
            logger.error("❌ No stored checksums found. Run generate command first.")
            return {}
        
        verification_results = {}
        critical_files = self.get_critical_files()
        
        logger.info(f"Verifying integrity of {len(critical_files)} files...")
        
        passed_files = 0
        failed_files = 0
        
        for file_path in critical_files:
            relative_path = str(file_path.relative_to(self.base_path))
            
            if relative_path not in stored_checksums:
                verification_results[relative_path] = {
                    'status': 'MISSING_CHECKSUM',
                    'passed': False,
                    'message': 'No stored checksum found'
                }
                failed_files += 1
                logger.warning(f"⚠️  No stored checksum for: {relative_path}")
                continue
            
            file_verification = self.verify_file_integrity(file_path, stored_checksums[relative_path])
            
            # File passes if ALL algorithms pass
            all_passed = all(file_verification.values())
            
            verification_results[relative_path] = {
                'status': 'PASSED' if all_passed else 'FAILED',
                'passed': all_passed,
                'algorithms': file_verification,
                'message': 'All checksums match' if all_passed else 'Checksum mismatch detected'
            }
            
            if all_passed:
                passed_files += 1
                logger.debug(f"✅ {relative_path}: PASSED")
            else:
                failed_files += 1
                logger.error(f"❌ {relative_path}: FAILED - {file_verification}")
        
        # Generate summary
        total_files = passed_files + failed_files
        success_rate = (passed_files / total_files * 100) if total_files > 0 else 0
        
        logger.info(f"📊 Verification Summary:")
        logger.info(f"   ✅ Passed: {passed_files}/{total_files} files ({success_rate:.1f}%)")
        logger.info(f"   ❌ Failed: {failed_files}/{total_files} files")
        
        if failed_files == 0:
            logger.info("🎉 ALL FILES PASSED INTEGRITY VERIFICATION")
        else:
            logger.error(f"💥 {failed_files} FILES FAILED INTEGRITY VERIFICATION")
        
        return verification_results
    
    def generate_integrity_report(self, verification_results: Dict[str, Dict[str, Union[bool, str]]]) -> None:
        """Generate comprehensive integrity report."""
        report = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'repository': str(self.base_path),
                'total_files': len(verification_results)
            },
            'summary': {
                'passed': sum(1 for r in verification_results.values() if r.get('passed', False)),
                'failed': sum(1 for r in verification_results.values() if not r.get('passed', False)),
                'success_rate': 0
            },
            'details': verification_results
        }
        
        total = report['summary']['passed'] + report['summary']['failed']
        if total > 0:
            report['summary']['success_rate'] = report['summary']['passed'] / total * 100
        
        try:
            with open(self.integrity_report, 'w') as f:
                json.dump(report, f, indent=2, sort_keys=True)
            
            logger.info(f"📄 Integrity report saved to: {self.integrity_report}")
            
        except Exception as e:
            logger.error(f"Error saving integrity report: {e}")
    
    def generate_checksums_command(self) -> bool:
        """Generate checksums command."""
        try:
            logger.info("🔒 Starting checksum generation...")
            
            checksums = self.generate_all_checksums()
            self.save_checksums(checksums)
            
            logger.info(f"✅ Successfully generated checksums for {len(checksums)} files")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during checksum generation: {e}")
            return False
    
    def verify_checksums_command(self) -> bool:
        """Verify checksums command."""
        try:
            logger.info("🔍 Starting integrity verification...")
            
            results = self.verify_all_files()
            self.generate_integrity_report(results)
            
            # Check if all files passed
            all_passed = all(r.get('passed', False) for r in results.values())
            
            if all_passed:
                logger.info("✅ VERIFICATION COMPLETED: PASSED")
            else:
                logger.error("❌ VERIFICATION COMPLETED: FAILED")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Error during verification: {e}")
            return False


def main():
    """Command-line interface for checksum operations."""
    parser = argparse.ArgumentParser(description="Iusmorfos Framework - File Integrity Management")
    
    parser.add_argument(
        'command',
        choices=['generate', 'verify'],
        help='Command to execute: generate checksums or verify integrity'
    )
    
    parser.add_argument(
        '--path',
        type=str,
        default='.',
        help='Base path to repository (default: current directory)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Initialize checksum generator
    generator = ChecksumGenerator(args.path)
    
    # Execute command
    if args.command == 'generate':
        success = generator.generate_checksums_command()
    elif args.command == 'verify':
        success = generator.verify_checksums_command()
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()