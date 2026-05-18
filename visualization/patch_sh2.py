import glob
import re

files = [
    'vis_occ_glob.sh',
    'vis_occ_online.sh',
    'vis_occ_rot.sh',
]

for f in files:
    with open(f, 'r') as file:
        c = file.read()
    
    # fix missing case 5
    if '5)' not in c and 'METHOD_NAME="Ours"' in c:
        c = re.sub(
            r'( {4}4\)\s+METHOD_NAME="Ours"\s+PCD_ROOT=\$\(python3 -c "import json; print\(json\.load\(open\(\'\$METHOD_CONFIG_FILE\'\)\)\[\'Ours\'\]\[\'pcd_root\'\]\)"\)\s+PCD_FOLD=\$\(python3 -c "import json; print\(json\.load\(open\(\'\$METHOD_CONFIG_FILE\'\)\)\[\'Ours\'\]\[\'pcd_fold\'\]\)"\)\s+OUTPUT_FOLDER=\$\(python3 -c "import json; print\(json\.load\(open\(\'\$METHOD_CONFIG_FILE\'\)\)\[\'Ours\'\]\[\'output_folder\'\]\)"\)\s+;;\s+)',
            r'\15)\n        METHOD_NAME="Ours_Colmap"\n        PCD_ROOT=$(python3 -c "import json; print(json.load(open(\'$METHOD_CONFIG_FILE\'))[\'Ours_Colmap\'][\'pcd_root\'])")\n        PCD_FOLD=$(python3 -c "import json; print(json.load(open(\'$METHOD_CONFIG_FILE\'))[\'Ours_Colmap\'][\'pcd_fold\'])")\n        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open(\'$METHOD_CONFIG_FILE\'))[\'Ours_Colmap\'][\'output_folder\'])")\n        ;;\n    ', c
        )
    with open(f, 'w') as file:
        file.write(c)

