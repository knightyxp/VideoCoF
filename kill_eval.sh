ps -ef | grep gpt_evaluation.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep gpt_evaluation.sh | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep clip_eval.sh | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep gpt_success_rate.sh | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep compute_clip_score.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep gpt_success_rate.py | grep -v grep | awk '{print $2}' | xargs kill -9
