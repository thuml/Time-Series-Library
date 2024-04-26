from anylearn import init_sdk, quick_train


# init_sdk('http://anylearn.nelbds.cn', 'DigitalLifeYZQiu', 'Qyz20020318!')
init_sdk('http://111.200.37.154:81/', 'DigitalLifeYZQiu', 'Qyz20020318!',disable_git=True)


for dataset in ['ETT-small-h1']:


    cmd = "sh ./scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh"

    print(cmd)
    task, _, _, _ = quick_train(
                        project_name='Parallel',
                        algorithm_cloud_name=f"PatchTST_{dataset}",
                        algorithm_local_dir="./",
                        algorithm_entrypoint=cmd,
                        algorithm_force_update=True,
                        algorithm_output="./outputs",
                        dataset_id=["DSET924f39a246e2bcba76feef284556"],
                        image_name="QUICKSTART_PYTORCH2.1.0_CUDA11.8_PYTHON3.11",
                        quota_group_request={
                            'name': "QGRPa1b75dd54023ab63d23d65261012",
                            'RTX-3090-unique': 1,
                            'CPU': 10,
                            'Memory': 50},
                        
                    )

