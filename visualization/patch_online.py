import re

with open('vis_occ_online.sh', 'r') as f:
    c = f.read()

# Frame list logic needs to be moved and modified
frame_list_old = """# Frame list: each scene has 30 frames, pcd_00000 ... pcd_00029
PCD_NAMES=()
for i in $(seq 0 29); do
    PCD_NAMES+=("pcd_$(printf '%05d' "$i")")
done"""

c = c.replace(frame_list_old, "")

insert_marker = 'if [ $scene_idx -lt 0 ] || [ $scene_idx -ge ${#PCD_SCENES[@]} ]; then\n    echo "❌ Invalid scene selection"\n    exit 1\nfi\n'
insert_idx = c.find(insert_marker)

if insert_idx != -1:
    insertion_point = insert_idx + len(insert_marker)
    new_frames_logic = """
# Frame list
if [ "$METHOD_NAME" = "Ours_Colmap" ]; then
    mapfile -t PCD_NAMES < <(ls -1 "$PCD_ROOT/$PCD_FOLD/${PCD_SCENES[scene_idx]}"/*.ply 2>/dev/null | xargs -n 1 basename | sed 's/\.ply$//')
else
    # Frame list: each scene has 30 frames, pcd_00000 ... pcd_00029
    PCD_NAMES=()
    for i in $(seq 0 29); do
        PCD_NAMES+=("pcd_$(printf '%05d' "$i")")
    done
fi
"""
    c = c[:insertion_point] + new_frames_logic + c[insertion_point:]

with open('vis_occ_online.sh', 'w') as f:
    f.write(c)

print("Updated vis_occ_online.sh")
