ps -ef | grep train_lora.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep train_1.3b.sh | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep train_joint_img_video_lora.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep train_joint_img_video_lora.sh | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep train_joint_img_cot_video_lora.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep train_joint_img_cot_video_lora.sh | grep -v grep | awk '{print $2}' | xargs kill -9
