# Zenodo Submission Guide for Iusmorfos Framework

## Overview
This guide provides step-by-step instructions for submitting the Iusmorfos framework to Zenodo for DOI generation and long-term preservation.

## Prerequisites
1. Zenodo account (https://zenodo.org)
2. ORCID identifier (recommended)
3. GitHub repository properly configured
4. All tests passing and documentation complete

## Automated Submission Steps

### 1. GitHub Integration (Recommended)
1. Log in to Zenodo and go to GitHub settings
2. Toggle ON the Iusmorfos repository for automatic preservation
3. Create a new release on GitHub:
   ```bash
   git tag -a v1.0.0 -m "First stable release"
   git push origin v1.0.0
   ```
4. Zenodo will automatically create a DOI for your release

### 2. Manual Upload Process
If GitHub integration is not available:

1. Create a repository archive:
   ```bash
   git archive --format=zip --output=iusmorfos-v1.0.0.zip HEAD
   ```

2. Log in to Zenodo and click "Upload"

3. Upload the archive file

4. Fill in metadata using the generated `zenodo_metadata.json`

5. Publish the record

## Metadata Configuration

The generated metadata includes:
- Complete bibliographic information
- Technical specifications and dependencies
- Validation and reproducibility details
- Keywords and subject classifications
- Git repository information
- Statistical robustness metrics

## Post-Submission Tasks

1. **Update Documentation**: Add DOI badge to README.md
   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)
   ```

2. **Update Citations**: Include DOI in all citations of the framework

3. **Link Publications**: Connect any related papers or datasets

4. **Community Engagement**: Share DOI in relevant scientific communities

## Continuous Integration

Set up automated DOI updates in CI/CD:
```yaml
- name: Update Zenodo on Release
  if: github.event_name == 'release'
  run: |
    # Trigger Zenodo webhook or API update
    curl -X POST "https://zenodo.org/api/records" 
         -H "Authorization: Bearer $ZENODO_TOKEN"
```

## Best Practices

1. **Version Management**: Use semantic versioning (v1.0.0, v1.1.0, etc.)
2. **Release Notes**: Include comprehensive release notes
3. **Breaking Changes**: Clearly document any breaking changes
4. **Migration Guides**: Provide upgrade instructions between versions

## Troubleshooting

### Common Issues
- **Metadata Validation**: Ensure all required fields are complete
- **File Size Limits**: Zenodo has upload limits (check current limits)
- **GitHub Integration**: May take time to sync, be patient

### Support Resources
- Zenodo Help: https://help.zenodo.org
- Zenodo Community: https://zenodo.org/communities
- GitHub Integration Docs: https://guides.github.com/activities/citable-code/

## Verification

After DOI assignment:
1. Verify the DOI resolves correctly
2. Check metadata accuracy on Zenodo record
3. Test download and installation from Zenodo archive
4. Update all documentation and citations

## Long-term Maintenance

1. **Regular Updates**: Submit new versions for significant changes
2. **Metadata Updates**: Keep contact information current
3. **Link Maintenance**: Ensure all related identifiers remain valid
4. **Community Feedback**: Monitor and respond to user feedback
