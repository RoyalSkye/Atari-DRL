
## Train NoisyNet DQN
```
$ python main.py --train --save_path <model_path> --noisy
```

## Resume Training
```
$ python main.py --train --resume --resume_path <previous_saved_model_path> --save_path <new_model_path> --best <previous_best_ep_score> --noisy
```

## Test model
```
$ python main.py --test --save_path <saved_model_path> --noisy
```

