from api import app, prewarm_preprocessing_runtime


prewarm_preprocessing_runtime()


if __name__ == "__main__":
    app.run()
