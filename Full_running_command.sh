#!/bin/bash

bash Prepare_ckpt.sh
bash -c "source scripts/config.sh && bash scripts/S1_dwpose_extract.sh" &
bash -c "source scripts/config.sh && bash scripts/S1_mask_extract.sh" &
bash -c "source scripts/config_smplerx.sh && bash scripts/S1_smplerx_extract.sh" &
bash -c "source scripts/config_smplerx.sh && bash scripts/S1_smplerx_extract1.sh" &
wait

bash -c "source scripts/config.sh && bash scripts/M3.5_hamer_extract.sh" &
wait

bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose1.sh" &
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose2.sh" &
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose3.sh" &
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose4.sh" &
wait
echo "Necessary files are prepared"

bash -c "source scripts/config.sh && bash scripts/F1_run_avatar.sh" &
wait
echo "All finished"
