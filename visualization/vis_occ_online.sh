#!/bin/bash

echo "🌐 Online Voxel Visualization Tool"
echo "=================================="
echo ""

# Method selection
echo "🔬 Select method:"
echo "  1. Label"
echo "  2. EmbodiedOcc"
echo "  3. Ours"
echo "  4. Ours_Colmap"
echo ""
read -p "Select method number: " method_choice

# Load method configuration from JSON
METHOD_CONFIG_FILE="visualization/method_config_online.json"

case $method_choice in
    1)
        METHOD_NAME="Label"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Label']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Label']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Label']['output_folder'])")
        ;;
    2)
        METHOD_NAME="EmbodiedOcc"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['EmbodiedOcc']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['EmbodiedOcc']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['EmbodiedOcc']['output_folder'])")
        ;;
    3)
        METHOD_NAME="Ours"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['output_folder'])")
        ;;
    4)
        METHOD_NAME="Ours_Colmap"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_Colmap']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_Colmap']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_Colmap']['output_folder'])")
        ;;
    *)
        echo "❌ Invalid method selection"
        exit 1
        ;;
esac

echo "✅ Selected method: $METHOD_NAME"
echo ""

# Visualization mode
echo "🎛️  Select visualization mode:"
echo "  1. Single frame"
echo "  2. Full sequence (30 frames + GIF)"
echo "  3. Single frame local"
echo ""
read -p "Select mode number: " mode_choice

case $mode_choice in
    1)
        VIS_MODE="single"
        ;;
    2)
        VIS_MODE="sequence"
        ;;
    3)
        VIS_MODE="single_local"
        ;;
    *)
        echo "❌ Invalid visualization mode selection"
        exit 1
        ;;
esac

echo "✅ Selected visualization mode: $VIS_MODE"
echo ""

# Configuration
PCD_EXT=".ply"

# Scene list
if [ "$METHOD_NAME" = "Ours_Colmap" ]; then
    mapfile -t PCD_SCENES < <(ls -1 "$PCD_ROOT/$PCD_FOLD" 2>/dev/null)
else
    PCD_SCENES=("scene0089_00" "scene0223_01" "scene0093_00" "scene0593_01" "scene0274_02" "scene0690_00" "scene0272_01" "scene0006_02"
            "scene0543_02" "scene0416_01" "scene0666_00" "scene0673_02" "scene0510_02" "scene0500_00" "scene0279_02" "scene0107_00")
fi



# Select scene
echo "📍 Available scenes:"
for i in "${!PCD_SCENES[@]}"; do
    echo "  $((i+1)). ${PCD_SCENES[i]}"
done
echo ""
read -p "Select scene number: " scene_choice
scene_idx=$((scene_choice-1))

if [ $scene_idx -lt 0 ] || [ $scene_idx -ge ${#PCD_SCENES[@]} ]; then
    echo "❌ Invalid scene selection"
    exit 1
fi

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

# Create output directory
OUTPUT_DIR="$PCD_ROOT/$OUTPUT_FOLDER/${PCD_SCENES[scene_idx]}"
mkdir -p "$OUTPUT_DIR"

show_3d="false"
include_frustum="false"
use_last_frame_baseline="true"
if [ "$VIS_MODE" = "single" ] || [ "$VIS_MODE" = "single_local" ]; then
    # Select frame only in single-frame mode
    echo ""
    echo "🎬 Available frames:"
    for i in "${!PCD_NAMES[@]}"; do
        echo "  $((i+1)). ${PCD_NAMES[i]}"
    done
    echo ""
    read -p "Select frame number: " name_choice
    name_idx=$((name_choice-1))

    if [ $name_idx -lt 0 ] || [ $name_idx -ge ${#PCD_NAMES[@]} ]; then
        echo "❌ Invalid frame selection"
        exit 1
    fi

    echo ""
    read -p "Show 3D interactive interface? (y/n) [default: n]: " show_3d_choice
    if [[ "$show_3d_choice" =~ ^[Yy]$ ]]; then
        show_3d="true"
    fi

    if [ "$VIS_MODE" = "single_local" ]; then
        include_frustum="true"
        use_last_frame_baseline="false"

        echo ""
        read -p "Include camera frustum? (y/n) [default: y]: " frustum_choice
        if [[ "$frustum_choice" =~ ^[Nn]$ ]]; then
            include_frustum="false"
        fi
    else
        echo ""
        read -p "Include camera frustum? (y/n) [default: n]: " frustum_choice
        if [[ "$frustum_choice" =~ ^[Yy]$ ]]; then
            include_frustum="true"
        fi

        echo ""
        read -p "Use last frame as baseline for scale/offset? (y/n) [default: y]: " baseline_choice
        if [[ "$baseline_choice" =~ ^[Nn]$ ]]; then
            use_last_frame_baseline="false"
        fi
    fi
else
    include_frustum="true"
    echo ""
    read -p "Include camera frustum in sequence? (y/n) [default: y]: " frustum_choice
    if [[ "$frustum_choice" =~ ^[Nn]$ ]]; then
        include_frustum="false"
    fi

    echo ""
    read -p "Use last frame as baseline for scale/offset? (y/n) [default: y]: " baseline_choice
    if [[ "$baseline_choice" =~ ^[Nn]$ ]]; then
        use_last_frame_baseline="false"
    fi
fi

# Ask whether to include RGB visualization beside PCD when rendering full sequence
include_rgb="false"
echo ""
read -p "Also visualize RGB images alongside PCD when making sequence GIF? (y/n) [default: n]: " rgb_choice
if [[ "$rgb_choice" =~ ^[Yy]$ ]]; then
    include_rgb="true"
fi

# GIF export tuning for README usage
GIF_MAX_WIDTH=960
GIF_FPS=4
GIF_MAX_COLORS=128

RUN_PCD_FOLD="$PCD_FOLD"
if [ "$VIS_MODE" = "single_local" ]; then
    RUN_PCD_FOLD="single_pred"
fi

echo ""
echo "🎯 Settings:"
echo "  Method: $METHOD_NAME"
echo "  Scene: ${PCD_SCENES[scene_idx]}"
echo "  Output: $OUTPUT_DIR"
echo "  Mode: $VIS_MODE"
if [ "$VIS_MODE" = "single" ]; then
    echo "  Frame: ${PCD_NAMES[name_idx]}"
elif [ "$VIS_MODE" = "single_local" ]; then
    echo "  Frame: ${PCD_NAMES[name_idx]}"
    echo "  Data fold: $RUN_PCD_FOLD"
else
    echo "  Frame range: ${PCD_NAMES[0]} ... ${PCD_NAMES[$(( ${#PCD_NAMES[@]} - 1 ))]}"
fi
echo "  3D Interface: $show_3d"
echo "  Camera Frustum: $include_frustum"
if [ "$VIS_MODE" = "single_local" ]; then
    echo "  Last-frame baseline: false (forced adaptive in local mode)"
else
    echo "  Last-frame baseline: $use_last_frame_baseline"
fi
echo "  View: Top-down fixed view"
echo "  Projection: Parallel (Fixed)"
echo ""

if [ "$VIS_MODE" = "single" ] || [ "$VIS_MODE" = "single_local" ]; then
    # Run Python script for a single frame
    frame_path="$PCD_ROOT/$RUN_PCD_FOLD/${PCD_SCENES[scene_idx]}/${PCD_NAMES[name_idx]}$PCD_EXT"
    if [ ! -f "$frame_path" ]; then
        echo "❌ Missing frame: $frame_path"
        exit 1
    fi

    is_local=$([ "$VIS_MODE" = "single_local" ] && echo "true" || echo "false")
    python3 visualization/vis_occ_online.py "$PCD_ROOT" "$RUN_PCD_FOLD" "${PCD_SCENES[scene_idx]}" "${PCD_NAMES[name_idx]}" "$PCD_EXT" "$OUTPUT_FOLDER" "$show_3d" "$include_frustum" "$use_last_frame_baseline" "$is_local"
else
    # Batch render the whole sequence, then build a GIF.
    if ! command -v xvfb-run &> /dev/null; then
        echo "❌ xvfb-run not found, required for offscreen sequence rendering"
        exit 1
    fi

    for frame_name in "${PCD_NAMES[@]}"; do
        frame_path="$PCD_ROOT/$RUN_PCD_FOLD/${PCD_SCENES[scene_idx]}/$frame_name$PCD_EXT"
        if [ ! -f "$frame_path" ]; then
            echo "❌ Missing frame: $frame_path"
            exit 1
        fi

        xvfb-run -a -s "-screen 0 1920x1200x24" \
            python3 visualization/vis_occ_online.py "$PCD_ROOT" "$RUN_PCD_FOLD" "${PCD_SCENES[scene_idx]}" "$frame_name" "$PCD_EXT" "$OUTPUT_FOLDER" "false" "$include_frustum" "$use_last_frame_baseline" "false"
    done

    scene_output_dir="$PCD_ROOT/$OUTPUT_FOLDER/${PCD_SCENES[scene_idx]}"
    gif_path="$scene_output_dir/${PCD_SCENES[scene_idx]}.gif"

    echo ""
    echo "🎞️  Creating GIF for the whole scene..."
    # If user requested RGB side-by-side, try to create combined images first (requires ImageMagick)
    if [ "$include_rgb" = "true" ]; then
        if command -v convert &> /dev/null && command -v identify &> /dev/null; then
            echo "🔗 Combining RGB + PCD images (left: RGB, right: PCD)"
            # Prefer iterating over the expected frame list to avoid missing frames
            # Load sequences JSON to query RGB paths
            SEQ_JSON="visualization/scene_image_sequences_all_30.json"
            if [ ! -f "$SEQ_JSON" ]; then
                echo "⚠️  $SEQ_JSON not found, falling back to direct path lookup"
                SEQ_JSON=""
            fi

            for frame_name in "${PCD_NAMES[@]}"; do
                # try variants of rendered filenames
                render_candidates=(
                    "$scene_output_dir/${frame_name}_t.png"
                    "$scene_output_dir/${frame_name}_frustum.png"
                    "$scene_output_dir/${frame_name}.png"
                    "$scene_output_dir/${frame_name}_render.png"
                )

                p=""
                for rc in "${render_candidates[@]}"; do
                    if [ -f "$rc" ]; then
                        p="$rc"
                        break
                    fi
                done

                if [ -z "$p" ]; then
                    echo "⚠️  Rendered image not found for ${frame_name}, skipping combine"
                    continue
                fi

                frame_idx="${frame_name#pcd_}"

                # Try to load RGB from sequences JSON first
                rgb_file=""
                if [ -n "$SEQ_JSON" ]; then
                    rgb_file=$(python3 visualization/query_rgb_path.py --scene "${PCD_SCENES[scene_idx]}" --frame_idx "$frame_idx" --json_path "$SEQ_JSON" 2>/dev/null)
                    if [ -n "$rgb_file" ]; then
                        echo "🔎 RGB path for ${frame_name}: $rgb_file"
                    fi
                fi


                # Fallback to direct path lookup if JSON query failed
                if [ -z "$rgb_file" ]; then
                    rgb_candidates=(
                        "data/occscannet/posed_images/${PCD_SCENES[scene_idx]}/${frame_idx}.jpg"
                        "data/occscannet/posed_images/${PCD_SCENES[scene_idx]}/${frame_idx}.png"
                        "data/occscannet/posed_imagse/${PCD_SCENES[scene_idx]}/${frame_idx}.jpg"
                        "data/occscannet/posed_imagse/${PCD_SCENES[scene_idx]}/${frame_idx}.png"
                    )

                    for c in "${rgb_candidates[@]}"; do
                        if [ -f "$c" ]; then
                            rgb_file="$c"
                            break
                        fi
                    done
                fi

                if [ -z "$rgb_file" ]; then
                    echo "⚠️  RGB not found for ${frame_name}, skipping combine"
                    continue
                fi

                # Use ffmpeg to combine (horizontal stack): RGB (left) + PCD (right)
                height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$p" 2>/dev/null)
                if [ -z "$height" ]; then
                    echo "⚠️  Could not get height for $p, skipping combine"
                    continue
                fi

                out_combined="$scene_output_dir/${frame_name}_combined_t.png"
                if [ -f "$out_combined" ]; then
                    # Combined already exists, skip
                    continue
                fi

                ffmpeg -y -i "$rgb_file" -i "$p" -filter_complex "[0:v]scale=-1:${height}[r];[r][1:v]hstack=inputs=2" "$out_combined" >/dev/null 2>&1
                if [ $? -eq 0 ]; then
                    echo "✓ Combined: $frame_name"
                else
                    echo "⚠️  ffmpeg failed for ${frame_name}, skipping"
                fi
            done

            # if we have combined images, use them for GIF, else fallback to PCD images
            if ls "$scene_output_dir"/*_combined_t.png >/dev/null 2>&1; then
                input_pattern="$scene_output_dir/*_combined_t.png"
            else
                input_pattern="$scene_output_dir/pcd_*_t.png"
            fi
        else
            echo "⚠️  ImageMagick not found — cannot auto-combine RGB. Proceeding with PCD-only GIF."
            input_pattern="$scene_output_dir/pcd_*_t.png"
        fi
    else
        input_pattern="$scene_output_dir/pcd_*_t.png"
    fi

    if command -v ffmpeg &> /dev/null; then
        ffmpeg -y -framerate "$GIF_FPS" -pattern_type glob -i "$input_pattern" \
            -filter_complex "[0:v]scale='min(${GIF_MAX_WIDTH},iw)':-2:flags=lanczos,fps=${GIF_FPS},split[s0][s1];[s0]palettegen=max_colors=${GIF_MAX_COLORS}[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
            "$gif_path" > /dev/null 2>&1
    elif command -v convert &> /dev/null; then
        convert -delay $((100 / GIF_FPS)) -loop 0 $input_pattern -resize "${GIF_MAX_WIDTH}x>" "$gif_path"
    else
        echo "❌ Neither ffmpeg nor ImageMagick found, cannot create GIF"
        exit 1
    fi

    if [ -f "$gif_path" ]; then
        echo "✅ GIF created: $gif_path"
    else
        echo "❌ GIF creation failed"
        exit 1
    fi
fi

echo ""
echo "✅ Done!"