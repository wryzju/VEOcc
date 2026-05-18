import os
import glob
import re

files = [
    '/home/komorebi/workspace/occ_gau_vistools/voxels/vis_occ_glob.sh',
    '/home/komorebi/workspace/occ_gau_vistools/voxels/vis_occ_online.sh',
    '/home/komorebi/workspace/occ_gau_vistools/voxels/vis_occ_rot.sh',
]

for fpath in files:
    with open(fpath, 'r') as f:
        content = f.read()

    # 1. Menu
    content = content.replace('echo "  4. Ours" \necho ""', 'echo "  4. Ours"\necho "  5. Ours_Colmap"\necho ""')
    content = content.replace('echo "  4. Ours"\necho ""', 'echo "  4. Ours"\necho "  5. Ours_Colmap"\necho ""')

    # 2. Case statement
    ours_case = """    4)
        METHOD_NAME="Ours"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['output_folder'])")
        ;;"""
    
    ours_colmap_case = """    4)
        METHOD_NAME="Ours"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['output_folder'])")
        ;;
    5)
        METHOD_NAME="Ours_Colmap"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_Colmap']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_Colmap']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_Colmap']['output_folder'])")
        ;;"""

    # If it's already there, replace might duplicate or won't match if it's there.
    if 'Ours_Colmap' not in content:
        content = content.replace(ours_case, ours_colmap_case)

    # 3. Scene list
    # Find PCD_SCENES=(...)
    scene_pattern = re.compile(r'(# Scene list\n)(PCD_SCENES=\([^)]+\))', re.MULTILINE)
    
    def scene_repl(m):
        return f'{m.group(1)}if [ "$METHOD_NAME" = "Ours_Colmap" ]; then\n    mapfile -t PCD_SCENES < <(ls -1 "$PCD_ROOT/$PCD_FOLD" 2>/dev/null)\nelse\n    {m.group(2)}\nfi'
    
    content = scene_pattern.sub(scene_repl, content)

    # 4. Frame list logic. 
    # Remove PCD_NAMES=(...) from where it is and put it after scene validation
    name_pattern = re.compile(r'# Frame list\s*\nPCD_NAMES=\([^)]+\)\s*\n', re.MULTILINE)
    
    # Extract the match first
    match = name_pattern.search(content)
    if match:
        pcd_names_str = match.group(0)
        content = content[:match.start()] + content[match.end():]
        
        # Now find where to insert it: after 'echo "❌ Invalid scene selection"\n    exit 1\nfi'
        insert_marker = 'if [ $scene_idx -lt 0 ] || [ $scene_idx -ge ${#PCD_SCENES[@]} ]; then\n    echo "❌ Invalid scene selection"\n    exit 1\nfi\n'
        
        insert_idx = content.find(insert_marker)
        if insert_idx != -1:
            insertion_point = insert_idx + len(insert_marker)
            
            pcd_names_var = pcd_names_str.replace('# Frame list', '').strip()
            
            new_frames_logic = f"""
# Frame list
if [ "$METHOD_NAME" = "Ours_Colmap" ]; then
    mapfile -t PCD_NAMES < <(ls -1 "$PCD_ROOT/$PCD_FOLD/${{PCD_SCENES[scene_idx]}}"/*.ply 2>/dev/null | xargs -n 1 basename | sed 's/\.ply$//')
else
    {pcd_names_var}
fi
"""
            content = content[:insertion_point] + new_frames_logic + content[insertion_point:]

    with open(fpath, 'w') as f:
        f.write(content)
    print("Patched:", fpath)

