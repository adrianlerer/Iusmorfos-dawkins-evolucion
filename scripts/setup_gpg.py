#!/usr/bin/env python3
"""
GPG signing and cryptographic verification setup for Iusmorfos repository.

This script sets up GPG key management, code signing, and cryptographic 
verification for maintaining research integrity and authenticity according
to academic publishing and open science standards.

Features:
- GPG key generation and management
- File and commit signing
- Signature verification workflows
- Integration with Git signing
- Academic identity verification
- Reproducible build signing

Usage:
    python scripts/setup_gpg.py [--generate-key] [--sign-files] [--verify] [--setup-git]
"""

import os
import sys
import subprocess
import argparse
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPGManager:
    """Comprehensive GPG key management and signing system"""
    
    def __init__(self, config_path: str = "config/gpg_config.yaml"):
        """
        Initialize GPG manager.
        
        Args:
            config_path: Path to GPG configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # GPG executable
        self.gpg_executable = self.find_gpg_executable()
        
        # File patterns to sign
        self.sign_patterns = [
            'README.md',
            'REPRODUCIBILITY.md',
            'CONTRIBUTING.md',
            'requirements.txt',
            'requirements.lock',
            'src/**/*.py',
            'scripts/**/*.py',
            'data/processed/**/*.csv',
            'notebooks/**/*.ipynb',
            'checksums.*',
            'ro-crate-metadata.json'
        ]
    
    def find_gpg_executable(self) -> str:
        """Find GPG executable on system"""
        for gpg_name in ['gpg', 'gpg2']:
            try:
                result = subprocess.run([gpg_name, '--version'], 
                                      capture_output=True, text=True, check=True)
                if result.returncode == 0:
                    logger.info(f"Found GPG: {gpg_name}")
                    return gpg_name
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        raise RuntimeError("GPG not found. Please install GnuPG.")
    
    def load_config(self) -> Dict:
        """Load GPG configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self.create_default_config()
        except Exception as e:
            logger.warning(f"Error loading GPG config: {e}. Using defaults.")
            return self.create_default_config()
    
    def create_default_config(self) -> Dict:
        """Create default GPG configuration"""
        config = {
            'key_settings': {
                'key_type': 'RSA',
                'key_length': 4096,
                'subkey_type': 'RSA',
                'subkey_length': 4096,
                'expire_date': '2y',
                'passphrase_required': True
            },
            'identity': {
                'real_name': 'Iusmorfos Research Team',
                'email': 'iusmorfos@research.org',
                'comment': 'GPG key for Iusmorfos reproducible research project'
            },
            'signing': {
                'sign_commits': True,
                'sign_tags': True,
                'sign_files': True,
                'signature_format': 'detached'
            },
            'verification': {
                'require_signatures': False,  # Set to True for strict verification
                'trusted_keys': [],
                'minimum_trust_level': 'marginal'
            }
        }
        
        # Save default config
        self.save_config(config)
        return config
    
    def save_config(self, config: Dict):
        """Save GPG configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, indent=2, sort_keys=False)
        
        logger.info(f"GPG configuration saved to {self.config_path}")
    
    def check_gpg_installation(self) -> bool:
        """Check if GPG is properly installed and accessible"""
        try:
            result = subprocess.run([self.gpg_executable, '--version'], 
                                  capture_output=True, text=True, check=True)
            
            version_info = result.stdout.split('\\n')[0]
            logger.info(f"GPG installation verified: {version_info}")
            return True
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("GPG is not installed or not accessible")
            return False
    
    def list_keys(self) -> List[Dict[str, str]]:
        """List available GPG keys"""
        try:
            result = subprocess.run([
                self.gpg_executable, '--list-keys', '--with-colons'
            ], capture_output=True, text=True, check=True)
            
            keys = []
            current_key = None
            
            for line in result.stdout.split('\\n'):
                if line.startswith('pub:'):
                    parts = line.split(':')
                    current_key = {
                        'type': 'public',
                        'trust': parts[1],
                        'length': parts[2],
                        'algorithm': parts[3],
                        'keyid': parts[4],
                        'creation': parts[5],
                        'expiry': parts[6],
                        'uid': '',
                        'fingerprint': ''
                    }
                elif line.startswith('uid:') and current_key:
                    parts = line.split(':')
                    if not current_key['uid']:  # Use first UID
                        current_key['uid'] = parts[9]
                elif line.startswith('fpr:') and current_key:
                    parts = line.split(':')
                    current_key['fingerprint'] = parts[9]
                    keys.append(current_key)
                    current_key = None
            
            return keys
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error listing GPG keys: {e}")
            return []
    
    def generate_key(self, passphrase: Optional[str] = None) -> Optional[str]:
        """
        Generate a new GPG key pair for the project.
        
        Args:
            passphrase: Key passphrase (if None, prompts user)
        
        Returns:
            Key ID if successful, None otherwise
        """
        identity = self.config['identity']
        key_settings = self.config['key_settings']
        
        # Create key generation script
        key_script = f\"\"\"
        Key-Type: {key_settings['key_type']}
        Key-Length: {key_settings['key_length']}
        Subkey-Type: {key_settings['subkey_type']}
        Subkey-Length: {key_settings['subkey_length']}
        Name-Real: {identity['real_name']}
        Name-Email: {identity['email']}
        Name-Comment: {identity['comment']}
        Expire-Date: {key_settings['expire_date']}
        %commit
        \"\"\"
        
        # Add passphrase if provided
        if passphrase:
            key_script = f"Passphrase: {passphrase}\\n" + key_script
        elif not key_settings.get('passphrase_required', True):
            key_script = "%no-protection\\n" + key_script
        
        logger.info("Generating GPG key pair...")
        logger.info(f"Name: {identity['real_name']}")
        logger.info(f"Email: {identity['email']}")
        logger.info(f"Key Type: {key_settings['key_type']} {key_settings['key_length']}")
        
        try:
            # Write key generation script to temporary file
            script_file = Path('.gpg_keygen_script')
            with open(script_file, 'w') as f:
                f.write(key_script)
            
            # Generate key
            result = subprocess.run([
                self.gpg_executable, '--batch', '--generate-key', str(script_file)
            ], capture_output=True, text=True, timeout=300)
            
            # Clean up script file
            script_file.unlink()
            
            if result.returncode == 0:
                # Extract key ID from output
                logger.info("GPG key pair generated successfully")
                
                # Get the newly created key ID
                keys = self.list_keys()
                if keys:
                    newest_key = max(keys, key=lambda k: k['creation'])
                    key_id = newest_key['keyid']
                    
                    logger.info(f"Key ID: {key_id}")
                    logger.info(f"Fingerprint: {newest_key['fingerprint']}")
                    
                    # Save key information to config
                    self.config['generated_key'] = {
                        'keyid': key_id,
                        'fingerprint': newest_key['fingerprint'],
                        'created_at': datetime.now().isoformat()
                    }
                    self.save_config(self.config)
                    
                    return key_id
                else:
                    logger.error("Could not find generated key")
                    return None
            else:
                logger.error(f"Key generation failed: {result.stderr}")
                return None
        
        except subprocess.TimeoutExpired:
            logger.error("Key generation timed out")
            return None
        except Exception as e:
            logger.error(f"Error generating key: {e}")
            return None
    
    def sign_file(self, file_path: Path, key_id: Optional[str] = None, 
                  detached: bool = True) -> bool:
        """
        Sign a file with GPG.
        
        Args:
            file_path: Path to file to sign
            key_id: GPG key ID to use (if None, uses default)
            detached: Create detached signature
        
        Returns:
            True if successful, False otherwise
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        cmd = [self.gpg_executable]
        
        # Add key ID if specified
        if key_id:
            cmd.extend(['-u', key_id])
        
        # Signature type
        if detached:
            cmd.extend(['--detach-sign', '--armor'])
            signature_ext = '.asc'
        else:
            cmd.extend(['--sign', '--armor'])
            signature_ext = '.gpg'
        
        cmd.extend(['--output', f"{file_path}{signature_ext}", str(file_path)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                logger.info(f"File signed: {file_path}")
                return True
            else:
                logger.error(f"Signing failed: {result.stderr}")
                return False
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error signing file {file_path}: {e}")
            return False
    
    def verify_signature(self, file_path: Path, signature_path: Optional[Path] = None) -> bool:
        """
        Verify a file signature.
        
        Args:
            file_path: Path to signed file
            signature_path: Path to signature file (for detached signatures)
        
        Returns:
            True if signature is valid, False otherwise
        """
        cmd = [self.gpg_executable, '--verify']
        
        if signature_path:
            cmd.extend([str(signature_path), str(file_path)])
        else:
            cmd.append(str(file_path))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Signature verified: {file_path}")
                return True
            else:
                logger.warning(f"Signature verification failed: {result.stderr}")
                return False
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error verifying signature: {e}")
            return False
    
    def sign_repository_files(self, key_id: Optional[str] = None) -> Dict[str, bool]:
        """
        Sign all critical repository files.
        
        Args:
            key_id: GPG key ID to use
        
        Returns:
            Dictionary mapping file paths to signing success status
        """
        from glob import glob
        
        results = {}
        
        logger.info("Signing repository files...")
        
        for pattern in self.sign_patterns:
            # Handle glob patterns
            if '*' in pattern:
                matches = glob(pattern, recursive=True)
                file_paths = [Path(match) for match in matches]
            else:
                file_paths = [Path(pattern)]
            
            for file_path in file_paths:
                if file_path.exists() and file_path.is_file():
                    success = self.sign_file(file_path, key_id)
                    results[str(file_path)] = success
                else:
                    logger.debug(f"Skipping non-existent file: {file_path}")
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"File signing complete: {successful}/{total} files signed successfully")
        
        return results
    
    def setup_git_signing(self, key_id: Optional[str] = None) -> bool:
        """
        Configure Git to use GPG signing for commits and tags.
        
        Args:
            key_id: GPG key ID to use for signing
        
        Returns:
            True if successful, False otherwise
        """
        if not key_id:
            # Try to get key from config
            generated_key = self.config.get('generated_key', {})
            key_id = generated_key.get('keyid')
            
            if not key_id:
                logger.error("No GPG key ID provided and none found in config")
                return False
        
        git_commands = [
            ['git', 'config', 'user.signingkey', key_id],
            ['git', 'config', 'commit.gpgsign', 'true'],
            ['git', 'config', 'tag.gpgsign', 'true'],
            ['git', 'config', 'gpg.program', self.gpg_executable]
        ]
        
        try:
            for cmd in git_commands:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.debug(f"Git config: {' '.join(cmd[2:])}")
            
            logger.info(f"Git signing configured with key: {key_id}")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error configuring Git signing: {e}")
            return False
    
    def export_public_key(self, key_id: str, output_file: str = 'public_key.asc') -> bool:
        """
        Export public key to file for sharing.
        
        Args:
            key_id: GPG key ID to export
            output_file: Output filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_file, 'w') as f:
                result = subprocess.run([
                    self.gpg_executable, '--armor', '--export', key_id
                ], stdout=f, capture_output=False, text=True, check=True)
            
            logger.info(f"Public key exported to {output_file}")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error exporting public key: {e}")
            return False
    
    def generate_verification_instructions(self) -> str:
        """Generate instructions for signature verification"""
        
        generated_key = self.config.get('generated_key', {})
        key_id = generated_key.get('keyid', 'KEY_ID')
        fingerprint = generated_key.get('fingerprint', 'FINGERPRINT')
        
        instructions = f\"\"\"# GPG Signature Verification for Iusmorfos
        
## Overview

This repository uses GPG signatures to ensure the authenticity and integrity of critical files. All signatures are created using the project's official GPG key.

## Project GPG Key Information

- **Key ID:** `{key_id}`
- **Fingerprint:** `{fingerprint}`
- **Purpose:** Signing Iusmorfos research artifacts and code

## Verification Steps

### 1. Import the Public Key

First, import the project's public key:

```bash
# Import from file (if available)
gpg --import public_key.asc

# Or import from keyserver
gpg --keyserver hkp://keyserver.ubuntu.com --recv-keys {key_id}
```

### 2. Verify File Signatures

For files with detached signatures (`.asc` files):

```bash
# Verify a specific file
gpg --verify filename.ext.asc filename.ext

# Verify all signed files
find . -name "*.asc" -exec gpg --verify {{}} \\;
```

### 3. Verify Git Commits and Tags

If Git signing is enabled:

```bash
# Verify the latest commit
git verify-commit HEAD

# Verify a specific tag
git verify-tag v1.0.0

# Show signature information
git log --show-signature
```

## Trust Levels

GPG uses trust levels to indicate confidence in key authenticity:

- **Unknown:** Key not verified
- **Marginal:** Some verification performed
- **Full:** Key fully verified and trusted
- **Ultimate:** Your own key or fully trusted

For research reproducibility, we recommend at least **marginal** trust.

## Troubleshooting

### "gpg: Can't check signature: No public key"

This means you haven't imported the signing key yet. Follow step 1 above.

### "gpg: WARNING: This key is not certified with a trusted signature!"

This is expected for new keys. You can increase trust by verifying the key fingerprint through an independent channel.

### Key Verification

To verify this is the authentic project key, compare the fingerprint:

```bash
gpg --fingerprint {key_id}
```

Expected fingerprint: `{fingerprint}`

## Security Notes

1. Always verify signatures before using repository contents in production
2. Check key fingerprints through multiple independent channels
3. Report any signature verification failures immediately
4. Keep GPG software updated to the latest version

## Contact

For questions about GPG signatures or key verification, please contact the project maintainers through the repository's issue tracker.

---

*Generated by Iusmorfos GPG Manager on {datetime.now().isoformat()}*
\"\"\"
        
        return instructions
    
    def create_signature_manifest(self) -> Dict[str, Dict]:
        """Create a manifest of all signed files and their signatures"""
        from glob import glob
        
        manifest = {
            'created_at': datetime.now().isoformat(),
            'gpg_key': self.config.get('generated_key', {}),
            'signed_files': {}
        }
        
        # Find all signature files
        signature_files = glob('**/*.asc', recursive=True)
        
        for sig_file in signature_files:
            sig_path = Path(sig_file)
            
            # Find corresponding original file
            orig_file = sig_path.with_suffix('')
            
            if orig_file.exists():
                # Verify signature
                is_valid = self.verify_signature(orig_file, sig_path)
                
                manifest['signed_files'][str(orig_file)] = {
                    'signature_file': str(sig_path),
                    'signature_valid': is_valid,
                    'file_size': orig_file.stat().st_size,
                    'signature_size': sig_path.stat().st_size,
                    'last_modified': datetime.fromtimestamp(orig_file.stat().st_mtime).isoformat()
                }
        
        # Save manifest
        with open('signature_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Signature manifest created with {len(manifest['signed_files'])} files")
        
        return manifest

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="GPG setup and signing for Iusmorfos repository")
    parser.add_argument('--generate-key', action='store_true',
                       help='Generate new GPG key pair')
    parser.add_argument('--sign-files', action='store_true',
                       help='Sign repository files')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing signatures')
    parser.add_argument('--setup-git', action='store_true',
                       help='Configure Git GPG signing')
    parser.add_argument('--export-key', action='store_true',
                       help='Export public key')
    parser.add_argument('--key-id', type=str,
                       help='Specific GPG key ID to use')
    parser.add_argument('--config', default='config/gpg_config.yaml',
                       help='Path to GPG configuration file')
    
    args = parser.parse_args()
    
    # Initialize GPG manager
    gpg_manager = GPGManager(args.config)
    
    # Check GPG installation
    if not gpg_manager.check_gpg_installation():
        logger.error("GPG installation check failed")
        return 1
    
    success = True
    
    if args.generate_key:
        logger.info("Generating new GPG key pair...")
        key_id = gpg_manager.generate_key()
        if key_id:
            logger.info(f"✅ Key generated successfully: {key_id}")
        else:
            logger.error("❌ Key generation failed")
            success = False
    
    if args.setup_git:
        logger.info("Setting up Git GPG signing...")
        if gpg_manager.setup_git_signing(args.key_id):
            logger.info("✅ Git signing configured successfully")
        else:
            logger.error("❌ Git signing configuration failed")
            success = False
    
    if args.sign_files:
        logger.info("Signing repository files...")
        results = gpg_manager.sign_repository_files(args.key_id)
        
        successful_signs = sum(1 for result in results.values() if result)
        total_files = len(results)
        
        if successful_signs == total_files:
            logger.info(f"✅ All {total_files} files signed successfully")
        else:
            logger.warning(f"⚠️ Only {successful_signs}/{total_files} files signed successfully")
            success = False
    
    if args.export_key:
        key_id = args.key_id or gpg_manager.config.get('generated_key', {}).get('keyid')
        if key_id:
            if gpg_manager.export_public_key(key_id):
                logger.info("✅ Public key exported successfully")
            else:
                logger.error("❌ Public key export failed")
                success = False
        else:
            logger.error("No key ID specified for export")
            success = False
    
    if args.verify:
        logger.info("Creating signature manifest and verification report...")
        manifest = gpg_manager.create_signature_manifest()
        
        valid_sigs = sum(1 for info in manifest['signed_files'].values() 
                        if info['signature_valid'])
        total_sigs = len(manifest['signed_files'])
        
        if valid_sigs == total_sigs:
            logger.info(f"✅ All {total_sigs} signatures verified successfully")
        else:
            logger.warning(f"⚠️ Only {valid_sigs}/{total_sigs} signatures are valid")
            success = False
    
    # Always generate verification instructions
    instructions = gpg_manager.generate_verification_instructions()
    
    with open('GPG_VERIFICATION.md', 'w') as f:
        f.write(instructions)
    
    logger.info("📝 GPG verification instructions saved to GPG_VERIFICATION.md")
    
    if success:
        logger.info("🎉 GPG setup completed successfully")
        return 0
    else:
        logger.error("💥 GPG setup completed with errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())