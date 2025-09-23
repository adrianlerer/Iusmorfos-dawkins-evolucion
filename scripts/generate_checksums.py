#!/usr/bin/env python3
"""
Comprehensive checksum generation and integrity verification system for Iusmorfos.

This script generates SHA-256, MD5, and SHA-512 checksums for all critical files
in the repository, ensuring data integrity and supporting reproducible research
verification according to FAIR and FORCE11 standards.

Usage:
    python scripts/generate_checksums.py [--verify] [--output checksums.json]
    
Features:
- Multi-hash verification (SHA-256, MD5, SHA-512)
- JSON and human-readable output formats
- File modification detection
- Integrity violation reporting
- Support for large file streaming
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('checksum_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class IntegrityVerifier:
    """Comprehensive file integrity verification system"""
    
    def __init__(self, root_dir: str = "."):
        """
        Initialize integrity verifier.
        
        Args:
            root_dir: Root directory for checksum operations
        """
        self.root_dir = Path(root_dir).resolve()
        self.hash_algorithms = {
            'sha256': hashlib.sha256,
            'md5': hashlib.md5,
            'sha512': hashlib.sha512
        }
        
        # Critical files that must be verified
        self.critical_files = [
            'src/*.py',
            'data/processed/*.csv',
            'notebooks/*.ipynb',
            'config/*.yaml',
            'requirements.txt',
            'requirements.lock',
            'Dockerfile',
            'README.md',
            'REPRODUCIBILITY.md',
            'CONTRIBUTING.md',
            'ro-crate-metadata.json'
        ]
        
        # Files to exclude from checksum generation
        self.exclude_patterns = [
            '.git/**',
            '__pycache__/**',
            '*.pyc',
            '.pytest_cache/**',
            'logs/**',
            'checksum_generation.log',
            'checksums.*',
            '.DS_Store',
            'Thumbs.db'
        ]
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Calculate hash for a single file using streaming to handle large files.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use ('sha256', 'md5', 'sha512')
        
        Returns:
            Hexadecimal hash string
        """
        if algorithm not in self.hash_algorithms:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_obj = self.hash_algorithms[algorithm]()
        
        try:
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(65536), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
        
        except (IOError, OSError) as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def calculate_multi_hash(self, file_path: Path) -> Dict[str, str]:
        """
        Calculate multiple hashes for a file simultaneously.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Dictionary mapping algorithm names to hash values
        """
        hash_objects = {alg: func() for alg, func in self.hash_algorithms.items()}
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    for hash_obj in hash_objects.values():
                        hash_obj.update(chunk)
            
            return {alg: hash_obj.hexdigest() for alg, hash_obj in hash_objects.items()}
        
        except (IOError, OSError) as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {alg: None for alg in self.hash_algorithms}
    
    def should_include_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be included in checksum generation.
        
        Args:
            file_path: Path to check
        
        Returns:
            True if file should be included
        """
        relative_path = file_path.relative_to(self.root_dir)
        
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if relative_path.match(pattern):
                return False
        
        # Include only regular files
        return file_path.is_file()
    
    def find_files_to_verify(self) -> List[Path]:
        """
        Find all files that should be included in integrity verification.
        
        Returns:
            List of file paths to verify
        """
        files_to_verify = []
        
        # Walk through all files in the repository
        for root, dirs, files in os.walk(self.root_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                file_path = Path(root) / file
                if self.should_include_file(file_path):
                    files_to_verify.append(file_path)
        
        return sorted(files_to_verify)
    
    def generate_checksums(self) -> Dict[str, Dict]:
        """
        Generate checksums for all relevant files.
        
        Returns:
            Dictionary containing file checksums and metadata
        """
        logger.info("Starting checksum generation...")
        
        files_to_verify = self.find_files_to_verify()
        checksums = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_files': len(files_to_verify),
                'hash_algorithms': list(self.hash_algorithms.keys()),
                'repository_root': str(self.root_dir),
                'generator_version': '1.0.0'
            },
            'files': {}
        }
        
        for i, file_path in enumerate(files_to_verify, 1):
            relative_path = str(file_path.relative_to(self.root_dir))
            logger.info(f"Processing {i}/{len(files_to_verify)}: {relative_path}")
            
            # Get file metadata
            stat = file_path.stat()
            
            # Calculate all hashes
            hashes = self.calculate_multi_hash(file_path)
            
            checksums['files'][relative_path] = {
                'hashes': hashes,
                'size_bytes': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'permissions': oct(stat.st_mode)[-3:]
            }
        
        logger.info(f"Checksum generation completed. {len(files_to_verify)} files processed.")
        return checksums
    
    def verify_checksums(self, checksum_file: str) -> Dict[str, List]:
        """
        Verify existing checksums against current file states.
        
        Args:
            checksum_file: Path to existing checksum file
        
        Returns:
            Dictionary with verification results
        """
        logger.info(f"Verifying checksums from {checksum_file}...")
        
        try:
            with open(checksum_file, 'r') as f:
                stored_checksums = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading checksum file: {e}")
            return {'errors': [f"Cannot load checksum file: {e}"]}
        
        results = {
            'verified': [],
            'modified': [],
            'missing': [],
            'new_files': [],
            'errors': []
        }
        
        stored_files = set(stored_checksums.get('files', {}).keys())
        current_files = {str(f.relative_to(self.root_dir)) for f in self.find_files_to_verify()}
        
        # Check for new files
        new_files = current_files - stored_files
        if new_files:
            results['new_files'] = list(new_files)
            logger.warning(f"Found {len(new_files)} new files not in original checksum")
        
        # Check for missing files
        missing_files = stored_files - current_files
        if missing_files:
            results['missing'] = list(missing_files)
            logger.warning(f"Found {len(missing_files)} missing files from original checksum")
        
        # Verify existing files
        for relative_path, stored_info in stored_checksums.get('files', {}).items():
            if relative_path in missing_files:
                continue
            
            file_path = self.root_dir / relative_path
            
            try:
                current_hashes = self.calculate_multi_hash(file_path)
                stored_hashes = stored_info.get('hashes', {})
                
                # Compare primary hash (SHA-256)
                if current_hashes.get('sha256') == stored_hashes.get('sha256'):
                    results['verified'].append(relative_path)
                else:
                    results['modified'].append({
                        'file': relative_path,
                        'stored_sha256': stored_hashes.get('sha256'),
                        'current_sha256': current_hashes.get('sha256')
                    })
                    logger.warning(f"File modified: {relative_path}")
            
            except Exception as e:
                error_msg = f"Error verifying {relative_path}: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        # Summary
        logger.info(f"Verification complete:")
        logger.info(f"  Verified: {len(results['verified'])} files")
        logger.info(f"  Modified: {len(results['modified'])} files")
        logger.info(f"  Missing: {len(results['missing'])} files")
        logger.info(f"  New: {len(results['new_files'])} files")
        logger.info(f"  Errors: {len(results['errors'])}")
        
        return results
    
    def save_checksums(self, checksums: Dict, output_file: str):
        """
        Save checksums to JSON file with human-readable backup.
        
        Args:
            checksums: Checksum dictionary
            output_file: Output filename
        """
        # Save JSON format
        json_file = output_file if output_file.endswith('.json') else f"{output_file}.json"
        with open(json_file, 'w') as f:
            json.dump(checksums, f, indent=2, sort_keys=True)
        
        logger.info(f"Checksums saved to {json_file}")
        
        # Save human-readable format
        txt_file = json_file.replace('.json', '.txt')
        with open(txt_file, 'w') as f:
            f.write(f"# Iusmorfos Repository Integrity Checksums\n")
            f.write(f"# Generated: {checksums['metadata']['generated_at']}\n")
            f.write(f"# Total files: {checksums['metadata']['total_files']}\n\n")
            
            for relative_path, info in sorted(checksums['files'].items()):
                f.write(f"File: {relative_path}\n")
                f.write(f"  Size: {info['size_bytes']:,} bytes\n")
                f.write(f"  Modified: {info['modified_time']}\n")
                for alg, hash_val in info['hashes'].items():
                    f.write(f"  {alg.upper()}: {hash_val}\n")
                f.write("\n")
        
        logger.info(f"Human-readable checksums saved to {txt_file}")
    
    def generate_verification_script(self, checksums: Dict, script_name: str = "verify_integrity.sh"):
        """
        Generate a shell script for independent verification.
        
        Args:
            checksums: Checksum dictionary
            script_name: Name of verification script
        """
        script_path = self.root_dir / script_name
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Iusmorfos Repository Integrity Verification Script\n")
            f.write(f"# Generated: {checksums['metadata']['generated_at']}\n")
            f.write("# Usage: bash verify_integrity.sh\n\n")
            
            f.write("echo \"🔐 Verifying Iusmorfos repository integrity...\"\n")
            f.write("FAILED=0\n\n")
            
            for relative_path, info in checksums['files'].items():
                sha256_hash = info['hashes']['sha256']
                f.write(f'echo "Verifying: {relative_path}"\n')
                f.write(f'COMPUTED=$(shasum -a 256 "{relative_path}" | cut -d" " -f1)\n')
                f.write(f'EXPECTED="{sha256_hash}"\n')
                f.write('if [ "$COMPUTED" != "$EXPECTED" ]; then\n')
                f.write(f'  echo "❌ FAILED: {relative_path}"\n')
                f.write('  echo "  Expected: $EXPECTED"\n')
                f.write('  echo "  Computed: $COMPUTED"\n')
                f.write('  FAILED=$((FAILED + 1))\n')
                f.write('else\n')
                f.write(f'  echo "✅ OK: {relative_path}"\n')
                f.write('fi\n\n')
            
            f.write('if [ $FAILED -eq 0 ]; then\n')
            f.write('  echo "🎉 All files verified successfully!"\n')
            f.write('  exit 0\n')
            f.write('else\n')
            f.write('  echo "💥 $FAILED files failed verification!"\n')
            f.write('  exit 1\n')
            f.write('fi\n')
        
        # Make script executable
        os.chmod(script_path, 0o755)
        logger.info(f"Verification script generated: {script_name}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Generate and verify file checksums for Iusmorfos repository")
    parser.add_argument('--verify', '-v', metavar='CHECKSUM_FILE', 
                       help='Verify existing checksums instead of generating new ones')
    parser.add_argument('--output', '-o', default='checksums.json',
                       help='Output filename for checksums (default: checksums.json)')
    parser.add_argument('--root-dir', '-r', default='.',
                       help='Repository root directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Initialize verifier
    verifier = IntegrityVerifier(args.root_dir)
    
    if args.verify:
        # Verification mode
        results = verifier.verify_checksums(args.verify)
        
        if results.get('errors'):
            logger.error("Verification completed with errors")
            return 1
        
        if results.get('modified') or results.get('missing'):
            logger.error("Integrity verification FAILED - files have been modified or are missing")
            return 1
        
        logger.info("✅ Integrity verification PASSED - all files are authentic")
        return 0
    
    else:
        # Generation mode
        checksums = verifier.generate_checksums()
        verifier.save_checksums(checksums, args.output)
        verifier.generate_verification_script(checksums)
        
        logger.info("✅ Checksum generation completed successfully")
        return 0

if __name__ == "__main__":
    sys.exit(main())