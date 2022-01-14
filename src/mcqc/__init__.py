from mcqc.consts import Consts
from mcqc.config import Config
import mcqc.nn
import mcqc.loss
import mcqc.datasets
import mcqc.models
import mcqc.training
import mcqc.utils



if __name__ == "__main__":
    from .main import app, main
    app.run(main)
