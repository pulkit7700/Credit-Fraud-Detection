

def som_(data):
    # Slicing
    import numpy as np
    import pandas as pd

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Scaling

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range=(0, 1))
    X = sc.fit_transform(X)
    from minisom import MiniSom
    # Training
    som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=1000)

    return som

def figure(som, X, y):
    # Visualizing the results
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # Creating the distance map
    p = ax.pcolor(som.distance_map().T)
    fig.colorbar(p)

    markers = ["o", "s"]
    colors = ["r", "g"]

    for i, x in enumerate(X):
        w = som.winner(x)
        ax.plot(
            w[0] + 0.5,
            w[1] + 0.5,
            markers[y[i]],
            markeredgecolor=colors[y[i]],
            markerfacecolor="None",
            markersize=10,
            markeredgewidth=2,
        )

    plt.show()