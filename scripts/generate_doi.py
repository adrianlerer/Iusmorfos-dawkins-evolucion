#!/usr/bin/env python3
"""
DOI generation and metadata management system for Iusmorfos repository.

This script generates DOI-ready metadata, creates Zenodo-compatible deposits,
and manages research object identification according to FAIR data principles
and academic publishing standards.

Features:
- DataCite metadata schema v4.4 compliance
- Zenodo API integration
- ORCID researcher identification
- Semantic versioning support
- Citation format generation
- RO-Crate integration

Usage:
    python scripts/generate_doi.py [--create-zenodo] [--update-metadata] [--preview]
"""

import os
import sys
import json
import yaml
import argparse
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DOIManager:
    """Comprehensive DOI generation and management system"""
    
    def __init__(self, config_path: str = "config/doi_config.yaml"):
        """
        Initialize DOI manager with configuration.
        
        Args:
            config_path: Path to DOI configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # DataCite metadata schema version
        self.datacite_version = "4.4"
        
        # Zenodo configuration
        self.zenodo_base_url = "https://zenodo.org/api"
        self.zenodo_sandbox_url = "https://sandbox.zenodo.org/api"
        
    def load_config(self) -> Dict[str, Any]:
        """Load DOI configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Create default configuration
                return self.create_default_config()
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
            return self.create_default_config()
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default DOI configuration"""
        config = {
            'project': {
                'title': 'Iusmorfos: Dawkins Biomorphs Applied to Legal Systems',
                'description': 'A comprehensive framework for analyzing legal system evolution using 9-dimensional iuspace methodology and cross-country validation.',
                'version': '1.0.0',
                'repository_url': 'https://github.com/yourusername/iusmorfos_public',
                'homepage': 'https://github.com/yourusername/iusmorfos_public',
                'license': 'MIT',
                'keywords': [
                    'legal systems',
                    'biomorphs',
                    'computational law',
                    'reproducible research',
                    'power-law analysis',
                    'cross-country validation',
                    'legal evolution',
                    'iuspace',
                    'FAIR data'
                ]
            },
            'authors': [
                {
                    'name': 'Research Team',
                    'affiliation': 'Research Institution',
                    'orcid': '0000-0000-0000-0000',
                    'email': 'contact@example.org'
                }
            ],
            'funding': [
                {
                    'agency': 'Research Funding Agency',
                    'grant_number': 'GRANT-2024-001',
                    'program': 'Computational Social Science Initiative'
                }
            ],
            'zenodo': {
                'use_sandbox': True,
                'access_token': None,  # Set via environment variable
                'communities': ['legal-tech', 'computational-social-science']
            },
            'datacite': {
                'publisher': 'Zenodo',
                'publication_year': datetime.now().year,
                'resource_type': 'Software',
                'language': 'en'
            }
        }
        
        # Save default config
        self.save_config(config)
        return config
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to YAML file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, indent=2, sort_keys=False)
        
        logger.info(f"Configuration saved to {self.config_path}")
    
    def generate_datacite_metadata(self) -> Dict[str, Any]:
        """
        Generate DataCite metadata schema v4.4 compliant metadata.
        
        Returns:
            DataCite metadata dictionary
        """
        project = self.config['project']
        authors = self.config['authors']
        datacite_config = self.config['datacite']
        
        # Generate unique identifier (will be replaced by actual DOI)
        identifier = f"10.5281/zenodo.{self.generate_placeholder_id()}"
        
        metadata = {
            'identifiers': [
                {
                    'identifier': identifier,
                    'identifierType': 'DOI'
                }
            ],
            'creators': [
                {
                    'creatorName': author['name'],
                    'affiliation': [author.get('affiliation', '')],
                    'nameIdentifiers': [
                        {
                            'nameIdentifier': author.get('orcid', ''),
                            'nameIdentifierScheme': 'ORCID',
                            'schemeURI': 'https://orcid.org/'
                        }
                    ] if author.get('orcid') else []
                }
                for author in authors
            ],
            'titles': [
                {
                    'title': project['title'],
                    'titleType': None
                }
            ],
            'publisher': datacite_config['publisher'],
            'publicationYear': datacite_config['publication_year'],
            'resourceType': {
                'resourceTypeGeneral': datacite_config['resource_type']
            },
            'descriptions': [
                {
                    'description': project['description'],
                    'descriptionType': 'Abstract'
                },
                {
                    'description': self.generate_technical_description(),
                    'descriptionType': 'TechnicalInfo'
                },
                {
                    'description': self.generate_methods_description(),
                    'descriptionType': 'Methods'
                }
            ],
            'subjects': [
                {'subject': keyword} for keyword in project['keywords']
            ],
            'language': datacite_config['language'],
            'version': project['version'],
            'rightsList': [
                {
                    'rights': 'MIT License',
                    'rightsURI': 'https://opensource.org/licenses/MIT'
                }
            ],
            'relatedIdentifiers': [
                {
                    'relatedIdentifier': project['repository_url'],
                    'relationType': 'IsSupplementTo',
                    'relatedIdentifierType': 'URL'
                }
            ],
            'fundingReferences': [
                {
                    'funderName': funding['agency'],
                    'awardNumber': funding['grant_number'],
                    'awardTitle': funding.get('program', '')
                }
                for funding in self.config.get('funding', [])
            ],
            'dates': [
                {
                    'date': datetime.now(timezone.utc).isoformat(),
                    'dateType': 'Created'
                },
                {
                    'date': datetime.now(timezone.utc).isoformat(),
                    'dateType': 'Updated'
                }
            ],
            'formats': [
                'application/zip',
                'text/csv',
                'application/json',
                'text/markdown',
                'application/x-python',
                'application/x-jupyter-notebook'
            ]
        }
        
        return metadata
    
    def generate_placeholder_id(self) -> str:
        """Generate placeholder Zenodo ID"""
        import hashlib
        content = f"{self.config['project']['title']}-{self.config['project']['version']}"
        hash_obj = hashlib.md5(content.encode())
        # Generate a 6-digit number from hash
        return str(int(hash_obj.hexdigest()[:8], 16) % 1000000)
    
    def generate_technical_description(self) -> str:
        """Generate technical description for metadata"""
        return """
        Technical Implementation Details:
        
        - Framework: 9-dimensional iuspace analysis methodology
        - Programming Language: Python 3.11+
        - Key Libraries: NumPy, Pandas, SciPy, Scikit-learn, Matplotlib
        - Statistical Methods: Bootstrap validation (1000+ iterations), power-law analysis (γ=2.3)
        - Cross-validation: 5-country validation (Argentina, Chile, South Africa, Sweden, India)
        - Reproducibility: Docker containerization, fixed random seeds, comprehensive testing
        - Data Format: CSV, JSON, RO-Crate metadata
        - Visualization: Interactive Plotly charts, Streamlit web interface
        - Documentation: Jupyter notebooks, API documentation, reproducibility guides
        - Quality Assurance: CI/CD pipeline, automated regression testing, checksum verification
        
        System Requirements:
        - CPU: Multi-core processor recommended for bootstrap analysis
        - RAM: 8GB minimum, 16GB recommended for full cross-country validation
        - Storage: 2GB for complete dataset and results
        - OS: Linux, macOS, Windows (via Docker)
        """
    
    def generate_methods_description(self) -> str:
        """Generate methodology description"""
        return """
        Methodology Overview:
        
        1. Legal Innovation Data Collection:
           - Primary dataset: 842 Argentine legal innovations (1990-2023)
           - Cross-country datasets: Chile, South Africa, Sweden, India
           - Data sources: Official legal databases, parliamentary records
        
        2. Iuspace Framework Application:
           - 9-dimensional gene analysis: complexity, scope, resistance, coherence, 
             adaptability, efficiency, legitimacy, sustainability, impact
           - Normalization to [0,1] range for cross-country comparison
           - Clustering analysis using K-means algorithm
        
        3. Statistical Validation:
           - Power-law distribution fitting (expected γ=2.3)
           - Bootstrap resampling with 1000+ iterations
           - Cross-validation across legal systems and cultural contexts
           - Kolmogorov-Smirnov goodness-of-fit testing
        
        4. Cultural Adaptation:
           - Hofstede cultural dimensions integration
           - Legal origin classification (civil law, common law, mixed systems)
           - Transferability analysis between countries
        
        5. Reproducibility Measures:
           - Fixed random seeds (seed=42)
           - Containerized execution environment
           - Comprehensive regression testing
           - Version-controlled dependencies
        """
    
    def generate_zenodo_metadata(self) -> Dict[str, Any]:
        """
        Generate Zenodo-specific metadata for deposit creation.
        
        Returns:
            Zenodo metadata dictionary
        """
        project = self.config['project']
        authors = self.config['authors']
        
        metadata = {
            'title': project['title'],
            'upload_type': 'software',
            'description': f"""
                <p><strong>{project['title']}</strong></p>
                
                <p>{project['description']}</p>
                
                <h3>Key Features:</h3>
                <ul>
                    <li>🧬 9-dimensional iuspace analysis methodology</li>
                    <li>⚡ Power-law validation (γ≈2.3) in legal citation networks</li>
                    <li>🌍 Cross-country validation across 5 legal systems</li>
                    <li>🔬 Bootstrap statistical robustness (1000+ iterations)</li>
                    <li>🎛️ Interactive Streamlit web interface</li>
                    <li>📊 Comprehensive Jupyter notebook analysis</li>
                    <li>🐳 Docker containerization for reproducibility</li>
                    <li>✅ CI/CD pipeline with automated testing</li>
                </ul>
                
                <h3>Reproducibility Standards:</h3>
                <ul>
                    <li>FAIR data principles compliance</li>
                    <li>FORCE11 reproducibility guidelines</li>
                    <li>Mozilla Open Science standards</li>
                    <li>ACM Artifact Review criteria</li>
                </ul>
                
                <h3>Usage:</h3>
                <p>See README.md and REPRODUCIBILITY.md for detailed installation and usage instructions.</p>
                
                <h3>Citation:</h3>
                <p>If you use this software in your research, please cite using the DOI provided.</p>
            """,
            'creators': [
                {
                    'name': author['name'],
                    'affiliation': author.get('affiliation', ''),
                    'orcid': author.get('orcid', '')
                }
                for author in authors
            ],
            'keywords': project['keywords'],
            'license': 'MIT',
            'version': project['version'],
            'language': 'eng',
            'related_identifiers': [
                {
                    'identifier': project['repository_url'],
                    'relation': 'isSupplementTo'
                }
            ],
            'communities': self.config['zenodo'].get('communities', [])
        }
        
        # Add funding information if available
        if self.config.get('funding'):
            funding_info = []
            for funding in self.config['funding']:
                funding_info.append(f"{funding['agency']} - {funding['grant_number']}")
            
            metadata['notes'] = f"Funding: {'; '.join(funding_info)}"
        
        return metadata
    
    def create_zenodo_deposit(self, use_sandbox: bool = True) -> Optional[Dict[str, Any]]:
        """
        Create a new Zenodo deposit.
        
        Args:
            use_sandbox: Use Zenodo sandbox environment
        
        Returns:
            Zenodo deposit response or None if failed
        """
        # Get access token from environment or config
        access_token = (
            os.environ.get('ZENODO_ACCESS_TOKEN') or 
            self.config['zenodo'].get('access_token')
        )
        
        if not access_token:
            logger.error("Zenodo access token not provided. Set ZENODO_ACCESS_TOKEN environment variable.")
            return None
        
        # Choose API endpoint
        base_url = self.zenodo_sandbox_url if use_sandbox else self.zenodo_base_url
        
        # Prepare metadata
        metadata = {'metadata': self.generate_zenodo_metadata()}
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        try:
            # Create deposit
            response = requests.post(
                f"{base_url}/deposit/depositions",
                json=metadata,
                headers=headers
            )
            
            if response.status_code == 201:
                deposit = response.json()
                logger.info(f"Zenodo deposit created successfully: {deposit['id']}")
                logger.info(f"Deposit URL: {deposit['links']['html']}")
                
                # Save deposit information
                self.save_deposit_info(deposit, use_sandbox)
                
                return deposit
            else:
                logger.error(f"Failed to create Zenodo deposit: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        
        except requests.RequestException as e:
            logger.error(f"Error creating Zenodo deposit: {e}")
            return None
    
    def save_deposit_info(self, deposit: Dict[str, Any], is_sandbox: bool):
        """Save Zenodo deposit information"""
        deposit_info = {
            'zenodo_id': deposit['id'],
            'pre_doi': deposit['metadata']['prereserve_doi']['doi'],
            'deposit_url': deposit['links']['html'],
            'api_url': deposit['links']['self'],
            'is_sandbox': is_sandbox,
            'created_at': datetime.now().isoformat(),
            'status': 'draft'
        }
        
        deposit_file = Path('zenodo_deposit.json')
        with open(deposit_file, 'w') as f:
            json.dump(deposit_info, f, indent=2)
        
        logger.info(f"Deposit information saved to {deposit_file}")
    
    def generate_citation_formats(self, doi: str = None) -> Dict[str, str]:
        """
        Generate various citation formats.
        
        Args:
            doi: DOI string (if None, uses placeholder)
        
        Returns:
            Dictionary of citation formats
        """
        project = self.config['project']
        authors = self.config['authors']
        
        if doi is None:
            doi = f"10.5281/zenodo.{self.generate_placeholder_id()}"
        
        # Author string
        if len(authors) == 1:
            author_str = authors[0]['name']
        elif len(authors) == 2:
            author_str = f"{authors[0]['name']} and {authors[1]['name']}"
        else:
            author_str = f"{authors[0]['name']} et al."
        
        year = self.config['datacite']['publication_year']
        
        citations = {
            'apa': f"{author_str} ({year}). {project['title']} (Version {project['version']}) [Computer software]. Zenodo. https://doi.org/{doi}",
            
            'mla': f"{author_str}. \"{project['title']}.\" Version {project['version']}, Zenodo, {year}, doi:{doi}.",
            
            'chicago': f"{author_str}. \"{project['title']}.\" Version {project['version']}. Zenodo, {year}. https://doi.org/{doi}.",
            
            'bibtex': f"""@software{{{project['title'].lower().replace(' ', '_')}_{year},
    title = {{{project['title']}}},
    author = {{{author_str}}},
    version = {{{project['version']}}},
    year = {{{year}}},
    publisher = {{Zenodo}},
    doi = {{{doi}}},
    url = {{https://doi.org/{doi}}}
}}""",
            
            'endnote': f"""%0 Computer Program
%A {author_str}
%T {project['title']}
%7 {project['version']}
%D {year}
%I Zenodo
%R {doi}
%U https://doi.org/{doi}""",
            
            'ris': f"""TY  - COMP
AU  - {author_str}
TI  - {project['title']}
PY  - {year}
PB  - Zenodo
DO  - {doi}
UR  - https://doi.org/{doi}
ER  -"""
        }
        
        return citations
    
    def update_ro_crate_metadata(self, doi: str = None):
        """Update RO-Crate metadata with DOI information"""
        ro_crate_file = Path('ro-crate-metadata.json')
        
        try:
            if ro_crate_file.exists():
                with open(ro_crate_file, 'r') as f:
                    ro_crate = json.load(f)
            else:
                logger.warning("RO-Crate metadata file not found")
                return
            
            # Update with DOI information
            if doi:
                # Add DOI to main dataset
                for entity in ro_crate.get('@graph', []):
                    if entity.get('@type') == 'Dataset' and 'iusmorfos' in entity.get('name', '').lower():
                        entity['identifier'] = f"https://doi.org/{doi}"
                        entity['sameAs'] = f"https://doi.org/{doi}"
                        break
            
            # Update modification date
            for entity in ro_crate.get('@graph', []):
                if entity.get('@id') == './':
                    entity['dateModified'] = datetime.now().isoformat()
                    break
            
            # Save updated RO-Crate
            with open(ro_crate_file, 'w') as f:
                json.dump(ro_crate, f, indent=2)
            
            logger.info(f"RO-Crate metadata updated with DOI: {doi}")
        
        except Exception as e:
            logger.error(f"Error updating RO-Crate metadata: {e}")
    
    def generate_doi_report(self) -> str:
        """Generate comprehensive DOI readiness report"""
        report_lines = [
            "# DOI Generation Report for Iusmorfos",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Project Version:** {self.config['project']['version']}",
            "",
            "## 📋 Metadata Validation",
            ""
        ]
        
        # Check required fields
        required_checks = [
            ("Project title", bool(self.config['project'].get('title'))),
            ("Project description", bool(self.config['project'].get('description'))),
            ("Authors information", len(self.config.get('authors', [])) > 0),
            ("Keywords", len(self.config['project'].get('keywords', [])) > 0),
            ("License specified", bool(self.config['project'].get('license'))),
            ("Repository URL", bool(self.config['project'].get('repository_url'))),
            ("Version number", bool(self.config['project'].get('version')))
        ]
        
        for check_name, passed in required_checks:
            status = "✅" if passed else "❌"
            report_lines.append(f"- {status} {check_name}")
        
        all_passed = all(passed for _, passed in required_checks)
        
        report_lines.extend([
            "",
            f"**Overall Status:** {'✅ Ready for DOI generation' if all_passed else '❌ Metadata incomplete'}",
            "",
            "## 🎯 DataCite Metadata Preview",
            "",
            "```json"
        ])
        
        # Add metadata preview
        datacite_metadata = self.generate_datacite_metadata()
        report_lines.append(json.dumps(datacite_metadata, indent=2))
        report_lines.extend([
            "```",
            "",
            "## 📚 Citation Formats Preview",
            ""
        ])
        
        # Add citation formats
        citations = self.generate_citation_formats()
        for format_name, citation in citations.items():
            report_lines.extend([
                f"### {format_name.upper()}",
                "",
                f"```",
                citation,
                "```",
                ""
            ])
        
        report_lines.extend([
            "## 🔗 Next Steps",
            "",
            "1. Review metadata completeness above",
            "2. Set ZENODO_ACCESS_TOKEN environment variable",
            "3. Run `python scripts/generate_doi.py --create-zenodo` to create deposit",
            "4. Upload files to Zenodo deposit",
            "5. Publish deposit to receive final DOI",
            "6. Update repository documentation with DOI",
            "",
            "---",
            "",
            f"*Report generated by Iusmorfos DOI Manager v1.0.0*"
        ])
        
        return "\n".join(report_lines)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Generate DOI and manage research object metadata")
    parser.add_argument('--create-zenodo', action='store_true',
                       help='Create Zenodo deposit')
    parser.add_argument('--sandbox', action='store_true', default=True,
                       help='Use Zenodo sandbox (default: True)')
    parser.add_argument('--update-metadata', action='store_true',
                       help='Update RO-Crate metadata with DOI')
    parser.add_argument('--preview', action='store_true',
                       help='Generate DOI readiness report')
    parser.add_argument('--config', default='config/doi_config.yaml',
                       help='Path to DOI configuration file')
    
    args = parser.parse_args()
    
    # Initialize DOI manager
    doi_manager = DOIManager(args.config)
    
    if args.preview:
        # Generate and save report
        report = doi_manager.generate_doi_report()
        report_file = Path('DOI_REPORT.md')
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        logger.info(f"DOI readiness report saved to {report_file}")
    
    elif args.create_zenodo:
        # Create Zenodo deposit
        deposit = doi_manager.create_zenodo_deposit(use_sandbox=args.sandbox)
        
        if deposit:
            pre_doi = deposit['metadata']['prereserve_doi']['doi']
            logger.info(f"Pre-reserved DOI: {pre_doi}")
            
            # Generate citation formats with pre-reserved DOI
            citations = doi_manager.generate_citation_formats(pre_doi)
            
            # Save citations
            citations_file = Path('CITATIONS.md')
            with open(citations_file, 'w') as f:
                f.write("# Citation Formats for Iusmorfos\n\n")
                for format_name, citation in citations.items():
                    f.write(f"## {format_name.upper()}\n\n```\n{citation}\n```\n\n")
            
            logger.info(f"Citation formats saved to {citations_file}")
            
            # Update RO-Crate if requested
            if args.update_metadata:
                doi_manager.update_ro_crate_metadata(pre_doi)
        
        else:
            logger.error("Failed to create Zenodo deposit")
            return 1
    
    elif args.update_metadata:
        # Load existing DOI from deposit file
        deposit_file = Path('zenodo_deposit.json')
        if deposit_file.exists():
            with open(deposit_file, 'r') as f:
                deposit_info = json.load(f)
            
            pre_doi = deposit_info.get('pre_doi')
            if pre_doi:
                doi_manager.update_ro_crate_metadata(pre_doi)
            else:
                logger.error("No DOI found in deposit information")
        else:
            logger.error("No Zenodo deposit information found. Create deposit first.")
    
    else:
        # Default: generate preview report
        report = doi_manager.generate_doi_report()
        print(report)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())