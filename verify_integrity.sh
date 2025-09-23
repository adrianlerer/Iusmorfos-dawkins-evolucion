#!/bin/bash
# Iusmorfos Repository Integrity Verification Script
# Generated: 2025-09-23T04:35:22.404926
# Usage: bash verify_integrity.sh

echo "🔐 Verifying Iusmorfos repository integrity..."
FAILED=0

echo "Verifying: ACHIEVEMENT_SUMMARY.md"
COMPUTED=$(shasum -a 256 "ACHIEVEMENT_SUMMARY.md" | cut -d" " -f1)
EXPECTED="2c6d95fe9d267f0a1cdb9287bae3ecc1d861146d70406cbec62629a2d0cb3f9c"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: ACHIEVEMENT_SUMMARY.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: ACHIEVEMENT_SUMMARY.md"
fi

echo "Verifying: CITATION.cff"
COMPUTED=$(shasum -a 256 "CITATION.cff" | cut -d" " -f1)
EXPECTED="e20561ea78dc8632140a16ea99e0ee0ed409c69464c35f741ac5485d6145ecb0"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: CITATION.cff"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: CITATION.cff"
fi

echo "Verifying: CONTRIBUTING.md"
COMPUTED=$(shasum -a 256 "CONTRIBUTING.md" | cut -d" " -f1)
EXPECTED="7b3e8381c1c8bdcd84fe716e9e07b1aa55650ebcaf03bd7444e28f79ec8b1ca0"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: CONTRIBUTING.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: CONTRIBUTING.md"
fi

echo "Verifying: Dockerfile"
COMPUTED=$(shasum -a 256 "Dockerfile" | cut -d" " -f1)
EXPECTED="8f9488017ec5aa5a8d9678e12487ac86282b8eeb25dd7d7acc6a906c74b66f30"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: Dockerfile"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: Dockerfile"
fi

echo "Verifying: LICENSE"
COMPUTED=$(shasum -a 256 "LICENSE" | cut -d" " -f1)
EXPECTED="031967299ec891870f37d0c05fa359007b40e24bd24a4890910542136763db1e"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: LICENSE"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: LICENSE"
fi

echo "Verifying: PRIORITY_6_COMPLETION.md"
COMPUTED=$(shasum -a 256 "PRIORITY_6_COMPLETION.md" | cut -d" " -f1)
EXPECTED="1c27159c3c2b2dbe1547e4b9baf20df5334ce33b70a60a12d865b757ce339561"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: PRIORITY_6_COMPLETION.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: PRIORITY_6_COMPLETION.md"
fi

echo "Verifying: README.md"
COMPUTED=$(shasum -a 256 "README.md" | cut -d" " -f1)
EXPECTED="f0a745ef6c3eae839e5495f2d6e3c4a5d8adeb830a9a598b581c791685cd4f3a"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: README.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: README.md"
fi

echo "Verifying: README_REPRODUCIBILITY.md"
COMPUTED=$(shasum -a 256 "README_REPRODUCIBILITY.md" | cut -d" " -f1)
EXPECTED="dfe731561714fb684d72e16493392e67eeb0c6e8ef46f124ac4c052c20087f6b"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: README_REPRODUCIBILITY.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: README_REPRODUCIBILITY.md"
fi

echo "Verifying: REPRODUCIBILITY.md"
COMPUTED=$(shasum -a 256 "REPRODUCIBILITY.md" | cut -d" " -f1)
EXPECTED="0c5f7ea051c79491c86d927ed7b495cb1e69ea77e7cc5bbffe45cb6a0a2af69b"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: REPRODUCIBILITY.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: REPRODUCIBILITY.md"
fi

echo "Verifying: SCIENTIFIC_DOCUMENTATION.md"
COMPUTED=$(shasum -a 256 "SCIENTIFIC_DOCUMENTATION.md" | cut -d" " -f1)
EXPECTED="6a2403b55e40a0d242a0658fe21b2b9aff9b85ce91c2ec065743710ef05f2c59"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: SCIENTIFIC_DOCUMENTATION.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: SCIENTIFIC_DOCUMENTATION.md"
fi

echo "Verifying: SECURITY.md"
COMPUTED=$(shasum -a 256 "SECURITY.md" | cut -d" " -f1)
EXPECTED="dbbffe9877d3dd75610593e61e881d3f9db54fdc6cb69a9fc50e6e5ab7d915bd"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: SECURITY.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: SECURITY.md"
fi

echo "Verifying: WORLD_CLASS_REPRODUCIBILITY_ACHIEVEMENT.md"
COMPUTED=$(shasum -a 256 "WORLD_CLASS_REPRODUCIBILITY_ACHIEVEMENT.md" | cut -d" " -f1)
EXPECTED="b63e38df76b5d7bd647b7f8ab4e3404aeb1e844a3da1e78efd6846f9497170be"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: WORLD_CLASS_REPRODUCIBILITY_ACHIEVEMENT.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: WORLD_CLASS_REPRODUCIBILITY_ACHIEVEMENT.md"
fi

echo "Verifying: app/streamlit_app.py"
COMPUTED=$(shasum -a 256 "app/streamlit_app.py" | cut -d" " -f1)
EXPECTED="f86a9ab9162d2991a7c8afe8ea2becfd9db432796ee153b0dfadd15485bd0c78"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: app/streamlit_app.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: app/streamlit_app.py"
fi

echo "Verifying: config/config.yaml"
COMPUTED=$(shasum -a 256 "config/config.yaml" | cut -d" " -f1)
EXPECTED="562650d97760bd6cddd6e6c82cbff16b497355db7b26221d06375c28f7d68aba"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: config/config.yaml"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: config/config.yaml"
fi

echo "Verifying: config/security_config.yaml"
COMPUTED=$(shasum -a 256 "config/security_config.yaml" | cut -d" " -f1)
EXPECTED="3d47d6e37ba09af9bb650e339f44b266d57753b864003a5d490e69f1bddf402c"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: config/security_config.yaml"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: config/security_config.yaml"
fi

echo "Verifying: data/evolution_cases.csv"
COMPUTED=$(shasum -a 256 "data/evolution_cases.csv" | cut -d" " -f1)
EXPECTED="51d2f0ed0307350130e12ae18f82da17ff568a83f8d68aa85dc1a8cdddfbb081"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: data/evolution_cases.csv"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: data/evolution_cases.csv"
fi

echo "Verifying: data/innovations_exported.csv"
COMPUTED=$(shasum -a 256 "data/innovations_exported.csv" | cut -d" " -f1)
EXPECTED="4e0ff4caa5348ad40991f062e9f202388bb48b8b48f281ca57879414c239dcdd"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: data/innovations_exported.csv"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: data/innovations_exported.csv"
fi

echo "Verifying: data/velocity_metrics.csv"
COMPUTED=$(shasum -a 256 "data/velocity_metrics.csv" | cut -d" " -f1)
EXPECTED="7cf476947bc2f293e880e208df8564f54f16efcef2bdae6f2f33fa3d61d63ca8"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: data/velocity_metrics.csv"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: data/velocity_metrics.csv"
fi

echo "Verifying: docs/CONTRIBUTING.md"
COMPUTED=$(shasum -a 256 "docs/CONTRIBUTING.md" | cut -d" " -f1)
EXPECTED="3c335314f9619e5a10ca54febdb2ace28bfb0966f68983d98cab7a8c11cda3ce"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: docs/CONTRIBUTING.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: docs/CONTRIBUTING.md"
fi

echo "Verifying: docs/REPLICATION.md"
COMPUTED=$(shasum -a 256 "docs/REPLICATION.md" | cut -d" " -f1)
EXPECTED="1d6f0cb6b6323bac1e9f25b28aedac450c443c503e394d0a12a6a5802f8175c2"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: docs/REPLICATION.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: docs/REPLICATION.md"
fi

echo "Verifying: environment.yml"
COMPUTED=$(shasum -a 256 "environment.yml" | cut -d" " -f1)
EXPECTED="eea8d6f2a453a6a6c97553dbe6bdb7a7f48ed9c92dfabe17c6f2820c61d897d1"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: environment.yml"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: environment.yml"
fi

echo "Verifying: notebooks/01_exploratory_data_analysis.ipynb"
COMPUTED=$(shasum -a 256 "notebooks/01_exploratory_data_analysis.ipynb" | cut -d" " -f1)
EXPECTED="f14e9b403bc33663dc62f2e4bf970a5ed8c7cf758347fb0011d5454e9a86bd97"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: notebooks/01_exploratory_data_analysis.ipynb"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: notebooks/01_exploratory_data_analysis.ipynb"
fi

echo "Verifying: notebooks/01_exploratory_data_analysis.py"
COMPUTED=$(shasum -a 256 "notebooks/01_exploratory_data_analysis.py" | cut -d" " -f1)
EXPECTED="217aca7bb047ee1155321f5e5ce318fe04df32b9c5ebab0856c86d1747657731"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: notebooks/01_exploratory_data_analysis.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: notebooks/01_exploratory_data_analysis.py"
fi

echo "Verifying: notebooks/02_external_validation_analysis.ipynb"
COMPUTED=$(shasum -a 256 "notebooks/02_external_validation_analysis.ipynb" | cut -d" " -f1)
EXPECTED="3f41cc9e49325d63b0fcc34ee25abdd5b99b36723d8d981ec9e43e6f223cd905"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: notebooks/02_external_validation_analysis.ipynb"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: notebooks/02_external_validation_analysis.ipynb"
fi

echo "Verifying: notebooks/02_statistical_diagnostics.ipynb"
COMPUTED=$(shasum -a 256 "notebooks/02_statistical_diagnostics.ipynb" | cut -d" " -f1)
EXPECTED="5a688762747b5bab247b58b0c6dfb37da5eeae15bd8095e5ab1f410fe5a88343"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: notebooks/02_statistical_diagnostics.ipynb"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: notebooks/02_statistical_diagnostics.ipynb"
fi

echo "Verifying: notebooks/03_cross_country_validation.ipynb"
COMPUTED=$(shasum -a 256 "notebooks/03_cross_country_validation.ipynb" | cut -d" " -f1)
EXPECTED="e4c2a0d2a54950c2aed6506bbc6bc20cec115ad122fc6937c7b3846a5813efcd"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: notebooks/03_cross_country_validation.ipynb"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: notebooks/03_cross_country_validation.ipynb"
fi

echo "Verifying: notebooks/Iusmorfos_Cloud_Analysis.ipynb"
COMPUTED=$(shasum -a 256 "notebooks/Iusmorfos_Cloud_Analysis.ipynb" | cut -d" " -f1)
EXPECTED="f5f4559ca88b460e741e3e51a576f67acfb24e1ae98b841503d4e3b4bf845e24"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: notebooks/Iusmorfos_Cloud_Analysis.ipynb"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: notebooks/Iusmorfos_Cloud_Analysis.ipynb"
fi

echo "Verifying: notebooks/Iusmorfos_Complete_Analysis.ipynb"
COMPUTED=$(shasum -a 256 "notebooks/Iusmorfos_Complete_Analysis.ipynb" | cut -d" " -f1)
EXPECTED="9dc1972122bdf9bb416d1c60c814ac36a61b83c5d4ea8e573a2d2a93e7c80b1a"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: notebooks/Iusmorfos_Complete_Analysis.ipynb"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: notebooks/Iusmorfos_Complete_Analysis.ipynb"
fi

echo "Verifying: paper/paper_biomorfos_legales_final.md"
COMPUTED=$(shasum -a 256 "paper/paper_biomorfos_legales_final.md" | cut -d" " -f1)
EXPECTED="2a86547829aee6ebb53324745d392a4ce2f5672a76220dc255fb810723e63c57"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: paper/paper_biomorfos_legales_final.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: paper/paper_biomorfos_legales_final.md"
fi

echo "Verifying: requirements.lock"
COMPUTED=$(shasum -a 256 "requirements.lock" | cut -d" " -f1)
EXPECTED="9a76598a079e7f3e0273f3ac260f1cc983d6e4c629dd97958c6c23faf1510f13"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: requirements.lock"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: requirements.lock"
fi

echo "Verifying: requirements.txt"
COMPUTED=$(shasum -a 256 "requirements.txt" | cut -d" " -f1)
EXPECTED="2f0e85b2cbbd30b35c4c33a71c041f37ff97340ff7164b60efbb922a39b2de00"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: requirements.txt"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: requirements.txt"
fi

echo "Verifying: results/biomorfos_legales_evolución.png"
COMPUTED=$(shasum -a 256 "results/biomorfos_legales_evolución.png" | cut -d" " -f1)
EXPECTED="368e28d81ad55cb9b9c771de04e924f3c62e4f5e3c920eda11d290989e707033"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: results/biomorfos_legales_evolución.png"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: results/biomorfos_legales_evolución.png"
fi

echo "Verifying: results/biomorfos_mejorado_20250921_054427.json"
COMPUTED=$(shasum -a 256 "results/biomorfos_mejorado_20250921_054427.json" | cut -d" " -f1)
EXPECTED="5365a1450ff8f9049bca5c21a655a004809d46da5a6ec69660584b6b127f33d4"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: results/biomorfos_mejorado_20250921_054427.json"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: results/biomorfos_mejorado_20250921_054427.json"
fi

echo "Verifying: results/validacion_piloto_20250921_054102.json"
COMPUTED=$(shasum -a 256 "results/validacion_piloto_20250921_054102.json" | cut -d" " -f1)
EXPECTED="caeade4b02c957f1032d09fbc50cba0d88d76aefc6b687ccd326b3936ce30115"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: results/validacion_piloto_20250921_054102.json"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: results/validacion_piloto_20250921_054102.json"
fi

echo "Verifying: ro-crate-metadata.json"
COMPUTED=$(shasum -a 256 "ro-crate-metadata.json" | cut -d" " -f1)
EXPECTED="372621bed62cb128715ac4c5a87bc074ce98d276694f4182c97ae713dc5cbb01"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: ro-crate-metadata.json"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: ro-crate-metadata.json"
fi

echo "Verifying: scripts/generate_checksums.py"
COMPUTED=$(shasum -a 256 "scripts/generate_checksums.py" | cut -d" " -f1)
EXPECTED="535c148794181d3361ecc10b59bb154bc0ef10220c8c679f3b383e19cfa4e859"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: scripts/generate_checksums.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: scripts/generate_checksums.py"
fi

echo "Verifying: scripts/generate_doi.py"
COMPUTED=$(shasum -a 256 "scripts/generate_doi.py" | cut -d" " -f1)
EXPECTED="62d6ba93b46f6a9e530bed6e965bb592595b7eb5213f372198ba3ca7d013eb5f"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: scripts/generate_doi.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: scripts/generate_doi.py"
fi

echo "Verifying: scripts/process_raw_data.py"
COMPUTED=$(shasum -a 256 "scripts/process_raw_data.py" | cut -d" " -f1)
EXPECTED="ea84a01e781eb8599260c0e40ee8f41357cb19a334a2fb35b4e3b3eafd046c9a"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: scripts/process_raw_data.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: scripts/process_raw_data.py"
fi

echo "Verifying: scripts/setup_gpg.py"
COMPUTED=$(shasum -a 256 "scripts/setup_gpg.py" | cut -d" " -f1)
EXPECTED="9742a17466188e3dd57ab5d65f8e684a493a443fa945a96e2b815470afff7e4f"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: scripts/setup_gpg.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: scripts/setup_gpg.py"
fi

echo "Verifying: security/INTEGRITY_REPORT.md"
COMPUTED=$(shasum -a 256 "security/INTEGRITY_REPORT.md" | cut -d" " -f1)
EXPECTED="973b0772023b458d93fdf5215dbbae977d4557ef4927c23702a16939f34d4b05"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: security/INTEGRITY_REPORT.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: security/INTEGRITY_REPORT.md"
fi

echo "Verifying: security/security_config.yaml"
COMPUTED=$(shasum -a 256 "security/security_config.yaml" | cut -d" " -f1)
EXPECTED="7b58e649afd39f7e833257905e44a3db280634219afa6ce1a7d901f9181a6a28"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: security/security_config.yaml"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: security/security_config.yaml"
fi

echo "Verifying: security/signatures/GPG_SETUP_INSTRUCTIONS.md"
COMPUTED=$(shasum -a 256 "security/signatures/GPG_SETUP_INSTRUCTIONS.md" | cut -d" " -f1)
EXPECTED="2d6e75c50af0ecd87b09c0077c1981c3bd4487feaea1d44251cf4a5fcfd5b78b"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: security/signatures/GPG_SETUP_INSTRUCTIONS.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: security/signatures/GPG_SETUP_INSTRUCTIONS.md"
fi

echo "Verifying: security/zenodo/ZENODO_SUBMISSION_GUIDE.md"
COMPUTED=$(shasum -a 256 "security/zenodo/ZENODO_SUBMISSION_GUIDE.md" | cut -d" " -f1)
EXPECTED="8f639209ac144c0fa4f45d37ef467ca7d6dada4f02cd222ecb5cac9ab101d3c6"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: security/zenodo/ZENODO_SUBMISSION_GUIDE.md"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: security/zenodo/ZENODO_SUBMISSION_GUIDE.md"
fi

echo "Verifying: security/zenodo/zenodo_metadata.json"
COMPUTED=$(shasum -a 256 "security/zenodo/zenodo_metadata.json" | cut -d" " -f1)
EXPECTED="12f0370d3cd9c2e0f5d18972f39e90377f86a600c0f1be399067049ff549c4e1"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: security/zenodo/zenodo_metadata.json"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: security/zenodo/zenodo_metadata.json"
fi

echo "Verifying: security/zenodo/zenodo_metadata.yaml"
COMPUTED=$(shasum -a 256 "security/zenodo/zenodo_metadata.yaml" | cut -d" " -f1)
EXPECTED="118b6e86166ef323d82964c298258fd694e1be94977c0d2e8d1a74c900bda797"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: security/zenodo/zenodo_metadata.yaml"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: security/zenodo/zenodo_metadata.yaml"
fi

echo "Verifying: security/zenodo_metadata.py"
COMPUTED=$(shasum -a 256 "security/zenodo_metadata.py" | cut -d" " -f1)
EXPECTED="c538d48876f8e2f17403a76457e4e92ca922084aceaa145da7e2ddf6aa43c223"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: security/zenodo_metadata.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: security/zenodo_metadata.py"
fi

echo "Verifying: src/baseline_models.py"
COMPUTED=$(shasum -a 256 "src/baseline_models.py" | cut -d" " -f1)
EXPECTED="d9d5fa4d37d92b8f52279ea8aac043d11c6be24efee7f7fc160d469712aea912"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: src/baseline_models.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: src/baseline_models.py"
fi

echo "Verifying: src/biomorfos_legales_dawkins.py"
COMPUTED=$(shasum -a 256 "src/biomorfos_legales_dawkins.py" | cut -d" " -f1)
EXPECTED="1ff656a0579e79d034bb7144bb6486120c55f9f778188bc7a057af92be616b21"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: src/biomorfos_legales_dawkins.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: src/biomorfos_legales_dawkins.py"
fi

echo "Verifying: src/biomorfos_legales_mejorado.py"
COMPUTED=$(shasum -a 256 "src/biomorfos_legales_mejorado.py" | cut -d" " -f1)
EXPECTED="f90b820983d1f4ea6dcfa05289b548937266cde7268c57dffe0fead050a2ef14"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: src/biomorfos_legales_mejorado.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: src/biomorfos_legales_mejorado.py"
fi

echo "Verifying: src/config.py"
COMPUTED=$(shasum -a 256 "src/config.py" | cut -d" " -f1)
EXPECTED="1be9692d1862c7bc8996b71dd0b1ea80a88043f6a65dc0c70b4f2e2e3a43e299"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: src/config.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: src/config.py"
fi

echo "Verifying: src/diagnostics.py"
COMPUTED=$(shasum -a 256 "src/diagnostics.py" | cut -d" " -f1)
EXPECTED="1aff79a9491bbc70812d3c81e12e45b01a3881910137612a70c85e5b224f2145"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: src/diagnostics.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: src/diagnostics.py"
fi

echo "Verifying: src/experimento_piloto_biomorfos.py"
COMPUTED=$(shasum -a 256 "src/experimento_piloto_biomorfos.py" | cut -d" " -f1)
EXPECTED="db9f61f6b40e7456ef7d1d9225b44e43e2df0d48a1b72706b3c2bc305256a367"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: src/experimento_piloto_biomorfos.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: src/experimento_piloto_biomorfos.py"
fi

echo "Verifying: src/external_validation.py"
COMPUTED=$(shasum -a 256 "src/external_validation.py" | cut -d" " -f1)
EXPECTED="c41321c17ef11ccb1d220fc4e05e2c086c0109c158d0d0da7966093f047269de"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: src/external_validation.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: src/external_validation.py"
fi

echo "Verifying: src/robustness.py"
COMPUTED=$(shasum -a 256 "src/robustness.py" | cut -d" " -f1)
EXPECTED="1d0a43635d621c6909810e176d319be1af2ad564c85d4098d1b2d2f837eabb0a"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: src/robustness.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: src/robustness.py"
fi

echo "Verifying: src/validacion_empirica_biomorfos.py"
COMPUTED=$(shasum -a 256 "src/validacion_empirica_biomorfos.py" | cut -d" " -f1)
EXPECTED="52fa74e293a4d89962ddfc7cf7742b3bdd1594851477780738a8e448af76262b"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: src/validacion_empirica_biomorfos.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: src/validacion_empirica_biomorfos.py"
fi

echo "Verifying: src/visualizacion_jusmorfos.py"
COMPUTED=$(shasum -a 256 "src/visualizacion_jusmorfos.py" | cut -d" " -f1)
EXPECTED="8931747b87e57812d318da3ddcb81215eea2d985a1d6a93d6574bc5bc9845237"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: src/visualizacion_jusmorfos.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: src/visualizacion_jusmorfos.py"
fi

echo "Verifying: streamlit_app.py"
COMPUTED=$(shasum -a 256 "streamlit_app.py" | cut -d" " -f1)
EXPECTED="8e9d1a165f1857c5bb6562004b793e7fe34163038e77950f813b6f78164f85c4"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: streamlit_app.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: streamlit_app.py"
fi

echo "Verifying: tests/test_regression.py"
COMPUTED=$(shasum -a 256 "tests/test_regression.py" | cut -d" " -f1)
EXPECTED="815230ecb5d050ec61597139d20410feaa65b4120efa5d49b8b92a26b7ad4f4f"
if [ "$COMPUTED" != "$EXPECTED" ]; then
  echo "❌ FAILED: tests/test_regression.py"
  echo "  Expected: $EXPECTED"
  echo "  Computed: $COMPUTED"
  FAILED=$((FAILED + 1))
else
  echo "✅ OK: tests/test_regression.py"
fi

if [ $FAILED -eq 0 ]; then
  echo "🎉 All files verified successfully!"
  exit 0
else
  echo "💥 $FAILED files failed verification!"
  exit 1
fi
