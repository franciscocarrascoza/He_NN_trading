# Repository Cleanup Plan - 2025-11-28

## Current State Analysis

### Space Usage
- **Total size**: ~4.3 GB
- **repo_backups/**: 3.7 GB (Git bundle - **DELETE**)
- **reports/**: 536 MB (training outputs - **KEEP but clean old**)
- **optuna_runs/**: 24 MB (old runs - **DELETE**)
- **logs**: 17 MB (optuna_v6_stage1.log - **ARCHIVE**)
- **databases**: ~6 MB (keep v6, delete v5)

### Critical Issues

#### 🚨 **SHOULD NOT BE IN GITHUB** (Add to .gitignore):
1. ✅ **repo_backups/** - Already in .gitignore, **DELETE FROM DISK**
2. ✅ **reports/** - Already in .gitignore (keep latest, clean old)
3. ❌ **optuna_hermite_v5.db** - Failed optimization DB - **DELETE**
4. ❌ **optuna_runs/** - NOT in .gitignore - **ADD & DELETE**
5. ❌ **Large log files** - NOT in .gitignore - **ADD

 to .gitignore**
6. ❌ **__pycache__** - Already in .gitignore but exists - **DELETE ALL**

#### 📁 **Virtual Environments**:
- NONE FOUND ✅ (good - using system anaconda env)

#### 📊 **Data Files**:
- No downloaded price data found in repo ✅
- calibrators/ - Generated model files (keep latest)
- plots/ - Generated visualizations (keep latest)
- pit/ - PIT test results (clean old)

### Cleanup Actions

#### Phase 1: Delete Space Wasters (Safe - Already Gitignored)
```bash
# 1. Delete 3.7GB git bundle backup
rm -rf repo_backups/

# 2. Clean old optuna runs (24 MB)
rm -rf optuna_runs/

# 3. Delete failed v5 database
rm optuna_hermite_v5.db

# 4. Delete all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

**Space saved**: ~3.7 GB

#### Phase 2: Archive Large Logs
```bash
# Move completed Stage 1 log to archive
mkdir -p docs/optimization_logs
gzip optuna_v6_stage1.log
mv optuna_v6_stage1.log.gz docs/optimization_logs/

# Keep Stage 2 log (still running)
```

**Space saved**: ~14 MB (compressed)

#### Phase 3: Organize Documentation
```bash
# Create docs structure
mkdir -p docs/{optuna,gui,fixes}

# Move Optuna docs
mv OPTUNA_*.md docs/optuna/
mv STAGE2_*.md docs/optuna/
mv SHARPE_VS_PVALUE_ANALYSIS.md docs/optuna/
mv stage1_top_models_for_stage2.csv docs/optuna/
mv LAST_BEST_PARAMETERS_OPT-V4 docs/optuna/

# Move GUI docs
mv GUI_*.md docs/gui/

# Move fix/change docs
mv *FIXES*.md *CHANGES*.md *ERRORS*.md *FIX_*.md *TESTING*.md *IMPLEMENTATION*.md docs/fixes/

# Keep in root: README.md, QUICK_START.md, SETUP_INSTRUCTIONS.md
```

#### Phase 4: Clean Old Reports
```bash
# Keep only latest summary and last 10 results
cd reports/

# Archive old results (keep last 10)
mkdir -p archive/
ls -t results_*.csv | tail -n +11 | xargs -I {} mv {} archive/ 2>/dev/null
ls -t results_*.md | tail -n +11 | xargs -I {} mv {} archive/ 2>/dev/null

# Clean old PIT files (keep last 20)
cd pit/
ls -t pit_*.csv | tail -n +21 | xargs rm -f 2>/dev/null

cd ../..
```

**Space saved**: ~400-500 MB

#### Phase 5: Update .gitignore
```bash
# Add missing entries
cat >> .gitignore <<'EOF'

# Optuna databases and logs
*.db
*.log
optuna_runs/

# Reports and generated files
reports/
*.csv
*.pkl

# Temporary files
*.tmp
*.bak
~*

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
EOF
```

### Files to Keep in Root

**Configuration**:
- config.yaml
- .gitignore
- requirements.txt (if exists)

**Documentation**:
- README.md
- QUICK_START.md
- SETUP_INSTRUCTIONS.md
- CLEANUP_PLAN.md (this file)

**Scripts** (active optimization):
- optuna_optimize_v6.py
- optuna_optimize_v6_stage2.py
- start_backend.sh
- diag.py

**Other**:
- best_config_optuna_v5.yaml
- best_config_optuna_v6_stage1.yaml (if exists)

### Post-Cleanup Structure

```
He_NN_trading/
├── README.md
├── QUICK_START.md
├── SETUP_INSTRUCTIONS.md
├── .gitignore (updated)
├── optuna_optimize_v6.py
├── optuna_optimize_v6_stage2.py
├── optuna_hermite_v6_stage1.db (5 MB - keep)
├── optuna_hermite_v6_stage2.db (growing - keep)
├── optuna_v6_stage2.log (growing - keep)
├── start_backend.sh
├── diag.py
├── best_config_*.yaml
│
├── src/ (source code - unchanged)
├── backend/ (unchanged)
├── ui/ (unchanged)
├── tests/ (unchanged)
├── scripts/ (unchanged)
│
├── docs/
│   ├── optuna/ (optuna documentation)
│   ├── gui/ (GUI documentation)
│   ├── fixes/ (historical fix docs)
│   └── optimization_logs/ (archived logs)
│
└── reports/ (latest 10 results + summary)
    ├── summary.json
    ├── results_latest.csv/md
    ├── plots/ (latest)
    ├── calibrators/ (latest)
    ├── pit/ (last 20)
    ├── predictions/ (latest)
    └── archive/ (older results)
```

### Expected Results

**Before**:
- Size: ~4.3 GB
- Root files: 46 files
- Documentation: Scattered across root

**After**:
- Size: ~150-200 MB
- Root files: ~15 files
- Documentation: Organized in docs/
- .gitignore: Properly configured

**GitHub repo size**: ~50-100 MB (excluding gitignored files)

### Safety Checks

✅ **Will NOT delete**:
- Source code (src/, backend/, ui/, tests/)
- Active databases (optuna_hermite_v6_stage1.db, optuna_hermite_v6_stage2.db)
- Running log (optuna_v6_stage2.log)
- Configuration files
- Active scripts

✅ **Will NOT interrupt**:
- Stage 2 optimization (still running)
- Any critical processes

### Validation Commands

```bash
# After cleanup, verify:
du -sh .                    # Should be ~150-200 MB
ls -lh *.db                 # Should show only v6 databases
ls -lh *.log                # Should show only v6_stage2.log
find . -name "__pycache__"  # Should return nothing
ls -lh docs/                # Should show organized structure
```

### Rollback Plan

If anything goes wrong:
1. Git bundle backup exists in repo_backups/ (before deletion)
2. All source code is in git
3. Databases can be regenerated (though Stage 1 takes ~3.5 hours)
4. Reports are regenerated on each training run

**Recommendation**: Create a final backup before cleanup:
```bash
# Optional: Create lightweight backup (no repo_backups/)
tar -czf ../He_NN_trading_before_cleanup_$(date +%Y%m%d).tar.gz \
  --exclude='repo_backups' \
  --exclude='__pycache__' \
  .
```
