from train_test import main_static, main_survey

dataset = 'static'

if __name__ == "__main__":
    if dataset == 'static':
        main_static()
    else:
        main_survey()