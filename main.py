import util
import LSTM


def main():
    data = util.load_data()
    models = LSTM.gen_models(data)

if __name__ == "__main__":
    main()
