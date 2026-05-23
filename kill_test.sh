ps -ef | grep predict_v2v_json.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep predict_v2v_cot_five_bench.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep predict_v2v_json_new.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep predict_v2v_cot_json.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep predict_v2v_dmd_cot_json.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep predict_v2v_json_openve_bench.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep predict_i2i_json.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep test_lora_1.3b.sh | grep -v grep | awk '{print $2}' | xargs kill -9