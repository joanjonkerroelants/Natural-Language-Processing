# Natural-Language-Processing
Assigment 1, 2 and 3 for NLP 

To get the repo open your terminal and type in: 
    ``` git clone {url} ```

## Reproducing the results from assignment 2:

Using *uv*:

1. CNN:

    ```uv run main.py neural cnn --max-len 146 --dropout {0 or 0.3}```

2. LSTM:

    ```uv run main.py neural lstm --max-len 146 {0 or 0.3}```

Using *pip*:

```pip install -r requirements.txt```

1. CNN:

    ```python main.py neural cnn --max-len 146 {0 or 0.3}```

2. LSTM:

    ```python main.py neural lstm --max-len 146 {0 or 0.3}```
