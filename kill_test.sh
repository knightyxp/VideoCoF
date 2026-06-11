ps -ef | grep predict_v2v_cot_json.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep predict_v2v_dmd_cot_json.py | grep -v grep | awk '{print $2}' | xargs kill -9
