# GPG Signing Setup for Iusmorfos Framework

## Overview
This document provides instructions for setting up GPG signing to ensure code authenticity and integrity.

## Installation

### Linux (Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install gnupg
```

### macOS
```bash
brew install gnupg
```

### Windows
Download and install GPG4Win from https://www.gpg4win.org/

## Key Generation

1. Generate a new GPG key pair:
```bash
gpg --full-generate-key
```

2. Choose RSA and RSA (default)
3. Use key size of 4096 bits
4. Set expiration (recommend 2 years)
5. Provide name and email (should match Git configuration)

## Export Public Key

```bash
# List keys to get KEY_ID
gpg --list-secret-keys --keyid-format=long

# Export public key
gpg --armor --export YOUR_KEY_ID > iusmorfos_public_key.asc
```

## Sign Repository Files

### Sign Git Commits
```bash
git config user.signingkey YOUR_KEY_ID
git config commit.gpgsign true
```

### Sign Release Tags
```bash
git tag -s v1.0.0 -m "Signed release v1.0.0"
```

### Sign Critical Files
```bash
# Sign individual files
gpg --detach-sign --armor requirements.lock
gpg --detach-sign --armor config/config.yaml

# Verify signatures
gpg --verify requirements.lock.asc requirements.lock
```

## Integration with CI/CD

Add GPG verification to your GitHub Actions workflow:

```yaml
- name: Import GPG key
  run: |
    echo "${{ secrets.GPG_PRIVATE_KEY }}" | gpg --import
    gpg --list-secret-keys

- name: Verify signatures
  run: |
    gpg --verify requirements.lock.asc requirements.lock
```

## Best Practices

1. **Key Security**: Store private keys securely, use strong passphrases
2. **Key Distribution**: Share public keys through trusted channels
3. **Regular Updates**: Rotate keys every 2-3 years
4. **Backup**: Maintain secure backups of key pairs
5. **Revocation**: Prepare revocation certificates

## Verification Commands

Users can verify authenticity using:

```bash
# Import public key
gpg --import iusmorfos_public_key.asc

# Verify file signature
gpg --verify file.txt.asc file.txt

# Verify git tag
git tag -v v1.0.0

# Verify commit
git log --show-signature
```
