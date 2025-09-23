#!/usr/bin/env python3
"""
Raw Data Processing for Iusmorfos
=================================

Processes raw legal innovation data into standardized formats for analysis.
Handles data validation, cleaning, and transformation with full provenance tracking.

Input: Raw CSV files from legal databases
Output: Standardized JSON files with validation metadata

Following FAIR principles and reproducibility best practices.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import hashlib
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import get_config


class LegalDataProcessor:
    """
    Processes raw legal innovation data with full validation and provenance.
    
    Features:
    - Data validation and quality checks
    - Standardized output formats
    - Provenance tracking
    - Statistical summaries
    - Error reporting
    """
    
    def __init__(self):
        """Initialize processor with configuration."""
        self.config = get_config()
        self.logger = logging.getLogger('iusmorfos.processor')
        
        self.processed_data = {}
        self.validation_report = {
            'processed_files': [],
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
    
    def process_innovations_dataset(self, 
                                   input_file: str,
                                   output_name: str = "innovations") -> Dict[str, Any]:
        """
        Process legal innovations dataset.
        
        Expected columns:
        - country: Country code (AR, CL, ZA, etc.)
        - year: Year of innovation
        - reform_type: Type of reform (constitutional, civil, criminal, etc.)
        - complexity_score: Complexity rating (1-10)
        - adoption_success: Success rate (0-1)
        - citation_count: Number of citations
        """
        self.logger.info(f"🔄 Processing innovations dataset: {input_file}")
        
        try:
            # Load raw data
            df = pd.read_csv(input_file)
            original_rows = len(df)
            
            # Validate required columns
            required_cols = ['country', 'year', 'reform_type', 'complexity_score', 
                           'adoption_success', 'citation_count']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}"
                self.logger.error(error_msg)
                self.validation_report['errors'].append(error_msg)
                return {}
            
            # Data cleaning and validation
            df_clean = self._clean_innovations_data(df)
            cleaned_rows = len(df_clean)
            
            # Calculate validation statistics
            validation_stats = self._calculate_validation_stats(df_clean, original_rows)
            
            # Transform to standardized format
            processed_data = self._transform_innovations_data(df_clean)
            
            # Add metadata
            processed_data['metadata'] = {
                'source_file': input_file,
                'processing_timestamp': datetime.now().isoformat(),
                'original_rows': original_rows,
                'cleaned_rows': cleaned_rows,
                'validation_passed': validation_stats['validation_passed'],
                'data_hash': self._calculate_data_hash(df_clean)
            }
            
            # Save processed data
            output_path = self.config.get_path('data_dir') / f"{output_name}_processed.json"
            self._save_processed_data(processed_data, output_path)
            
            self.processed_data[output_name] = processed_data
            self.validation_report['processed_files'].append({
                'name': output_name,
                'input_file': input_file,
                'output_file': str(output_path),
                'statistics': validation_stats
            })
            
            self.logger.info(f"✅ Processed {output_name}: {cleaned_rows} records")
            return processed_data
            
        except Exception as e:
            error_msg = f"Failed to process {input_file}: {str(e)}"
            self.logger.error(error_msg)
            self.validation_report['errors'].append(error_msg)
            return {}
    
    def _clean_innovations_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate innovations data."""
        df_clean = df.copy()
        
        # Remove duplicates
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_len:
            self.logger.warning(f"Removed {initial_len - len(df_clean)} duplicate records")
        
        # Validate year range
        current_year = datetime.now().year
        valid_year_mask = (df_clean['year'] >= 1800) & (df_clean['year'] <= current_year)
        invalid_years = len(df_clean) - valid_year_mask.sum()
        if invalid_years > 0:
            self.logger.warning(f"Removing {invalid_years} records with invalid years")
            df_clean = df_clean[valid_year_mask]
        
        # Validate complexity scores (should be 1-10)
        valid_complexity_mask = (df_clean['complexity_score'] >= 1) & (df_clean['complexity_score'] <= 10)
        invalid_complexity = len(df_clean) - valid_complexity_mask.sum()
        if invalid_complexity > 0:
            self.logger.warning(f"Removing {invalid_complexity} records with invalid complexity scores")
            df_clean = df_clean[valid_complexity_mask]
        
        # Validate adoption success (should be 0-1)
        valid_adoption_mask = (df_clean['adoption_success'] >= 0) & (df_clean['adoption_success'] <= 1)
        invalid_adoption = len(df_clean) - valid_adoption_mask.sum()
        if invalid_adoption > 0:
            self.logger.warning(f"Removing {invalid_adoption} records with invalid adoption success rates")
            df_clean = df_clean[valid_adoption_mask]
        
        # Validate citation counts (non-negative)
        valid_citation_mask = df_clean['citation_count'] >= 0
        invalid_citations = len(df_clean) - valid_citation_mask.sum()
        if invalid_citations > 0:
            self.logger.warning(f"Removing {invalid_citations} records with negative citation counts")
            df_clean = df_clean[valid_citation_mask]
        
        return df_clean
    
    def _calculate_validation_stats(self, df_clean: pd.DataFrame, original_rows: int) -> Dict[str, Any]:
        """Calculate validation statistics."""
        cleaned_rows = len(df_clean)
        data_quality = cleaned_rows / original_rows if original_rows > 0 else 0
        
        stats = {
            'original_rows': original_rows,
            'cleaned_rows': cleaned_rows,
            'data_quality_ratio': data_quality,
            'validation_passed': data_quality >= 0.8,  # At least 80% data retained
            'country_coverage': df_clean['country'].nunique(),
            'year_range': [int(df_clean['year'].min()), int(df_clean['year'].max())],
            'mean_complexity': float(df_clean['complexity_score'].mean()),
            'mean_adoption_success': float(df_clean['adoption_success'].mean()),
            'total_citations': int(df_clean['citation_count'].sum())
        }
        
        return stats
    
    def _transform_innovations_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Transform data to standardized Iusmorfos format."""
        
        # Group by country and reform type for analysis
        country_stats = {}
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            
            country_stats[country] = {
                'total_innovations': len(country_data),
                'reform_types': country_data['reform_type'].value_counts().to_dict(),
                'complexity_distribution': {
                    'mean': float(country_data['complexity_score'].mean()),
                    'std': float(country_data['complexity_score'].std()),
                    'min': int(country_data['complexity_score'].min()),
                    'max': int(country_data['complexity_score'].max())
                },
                'adoption_success_rate': float(country_data['adoption_success'].mean()),
                'citation_network': {
                    'total_citations': int(country_data['citation_count'].sum()),
                    'mean_citations': float(country_data['citation_count'].mean()),
                    'citation_distribution': country_data['citation_count'].describe().to_dict()
                }
            }
        
        # Create time series data
        time_series = df.groupby('year').agg({
            'complexity_score': ['mean', 'std', 'count'],
            'adoption_success': 'mean',
            'citation_count': 'sum'
        }).round(3).to_dict()
        
        # Prepare evolution data for Iusmorfos analysis
        evolution_data = []
        for _, row in df.iterrows():
            evolution_data.append({
                'country': row['country'],
                'year': int(row['year']),
                'reform_type': row['reform_type'],
                'iuspace_coordinates': {
                    'complexity': float(row['complexity_score']),
                    'adoption': float(row['adoption_success']),
                    'citations': float(row['citation_count'])
                },
                'fitness_score': self._calculate_fitness_score(row)
            })
        
        return {
            'country_statistics': country_stats,
            'time_series': time_series,
            'evolution_data': evolution_data,
            'summary': {
                'total_records': len(df),
                'countries_covered': len(country_stats),
                'year_span': int(df['year'].max() - df['year'].min()),
                'reform_types': df['reform_type'].nunique()
            }
        }
    
    def _calculate_fitness_score(self, row: pd.Series) -> float:
        """Calculate fitness score for a legal innovation."""
        # Normalized fitness combining complexity, adoption success, and citation impact
        complexity_norm = row['complexity_score'] / 10.0  # Normalize to 0-1
        adoption = row['adoption_success']  # Already 0-1
        
        # Citation impact (log-normalized to handle power-law distribution)
        citation_impact = np.log1p(row['citation_count']) / 10.0 if row['citation_count'] > 0 else 0
        citation_impact = min(citation_impact, 1.0)  # Cap at 1.0
        
        # Weighted fitness score
        fitness = (0.3 * complexity_norm + 0.4 * adoption + 0.3 * citation_impact)
        return round(fitness, 4)
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of cleaned data for integrity checking."""
        data_string = df.to_string(index=False)
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def _save_processed_data(self, data: Dict[str, Any], output_path: Path) -> None:
        """Save processed data with metadata."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Saved processed data: {output_path}")
    
    def process_crisis_dataset(self, input_file: str, output_name: str = "crises") -> Dict[str, Any]:
        """
        Process crisis periods dataset.
        
        Expected columns:
        - country: Country code
        - start_year: Crisis start year  
        - end_year: Crisis end year
        - crisis_type: Type of crisis (economic, political, social)
        - severity: Crisis severity (1-10)
        - recovery_time: Time to recovery in years
        """
        self.logger.info(f"🔄 Processing crisis dataset: {input_file}")
        
        try:
            df = pd.read_csv(input_file)
            original_rows = len(df)
            
            # Basic validation and cleaning
            required_cols = ['country', 'start_year', 'end_year', 'crisis_type', 'severity']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                error_msg = f"Missing required columns in crisis data: {missing_cols}"
                self.logger.error(error_msg)
                self.validation_report['errors'].append(error_msg)
                return {}
            
            # Clean data
            df_clean = df.copy()
            df_clean = df_clean.dropna(subset=required_cols)
            
            # Validate year ranges
            valid_years = (df_clean['start_year'] <= df_clean['end_year'])
            df_clean = df_clean[valid_years]
            
            # Transform to standardized format
            processed_data = {
                'crisis_periods': df_clean.to_dict('records'),
                'metadata': {
                    'source_file': input_file,
                    'processing_timestamp': datetime.now().isoformat(),
                    'original_rows': original_rows,
                    'cleaned_rows': len(df_clean),
                    'data_hash': self._calculate_data_hash(df_clean)
                }
            }
            
            # Save processed data
            output_path = self.config.get_path('data_dir') / f"{output_name}_processed.json"
            self._save_processed_data(processed_data, output_path)
            
            self.processed_data[output_name] = processed_data
            self.logger.info(f"✅ Processed {output_name}: {len(df_clean)} records")
            return processed_data
            
        except Exception as e:
            error_msg = f"Failed to process crisis data {input_file}: {str(e)}"
            self.logger.error(error_msg)
            self.validation_report['errors'].append(error_msg)
            return {}
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        self.validation_report['summary'] = {
            'total_files_processed': len(self.validation_report['processed_files']),
            'total_errors': len(self.validation_report['errors']),
            'total_warnings': len(self.validation_report['warnings']),
            'overall_success': len(self.validation_report['errors']) == 0
        }
        
        # Save validation report
        report_path = self.config.get_path('results_dir') / f"data_validation_report_{self.config.timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 Validation report saved: {report_path}")
        return self.validation_report


def main():
    """Main processing function."""
    print("🧬 Iusmorfos Data Processing Pipeline")
    print("=" * 50)
    
    processor = LegalDataProcessor()
    
    # Process sample datasets (these would be real files in production)
    data_dir = processor.config.get_path('data_dir')
    
    # Create sample data files if they don't exist (for testing)
    sample_innovations_file = data_dir / "sample_innovations.csv"
    if not sample_innovations_file.exists():
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'country': ['AR'] * 50 + ['CL'] * 30 + ['ZA'] * 20,
            'year': np.random.randint(2000, 2024, 100),
            'reform_type': np.random.choice(['constitutional', 'civil', 'criminal', 'administrative'], 100),
            'complexity_score': np.random.randint(1, 11, 100),
            'adoption_success': np.random.uniform(0, 1, 100),
            'citation_count': np.random.poisson(5, 100)
        })
        
        sample_data.to_csv(sample_innovations_file, index=False)
        print(f"📝 Created sample data: {sample_innovations_file}")
    
    # Process datasets
    innovations_data = processor.process_innovations_dataset(
        str(sample_innovations_file), 
        "argentina_innovations"
    )
    
    # Generate validation report
    validation_report = processor.generate_validation_report()
    
    # Print summary
    print(f"\n📊 Processing Summary:")
    print(f"Files processed: {validation_report['summary']['total_files_processed']}")
    print(f"Errors: {validation_report['summary']['total_errors']}")
    print(f"Warnings: {validation_report['summary']['total_warnings']}")
    print(f"Success: {validation_report['summary']['overall_success']}")
    
    if innovations_data:
        print(f"\n🇦🇷 Argentina Innovations:")
        print(f"Countries: {innovations_data['summary']['countries_covered']}")
        print(f"Records: {innovations_data['summary']['total_records']}")
        print(f"Year span: {innovations_data['summary']['year_span']} years")


if __name__ == "__main__":
    main()