def parse_layers(layer_list):
    rez=""
    dense_layers = list()
    dropout_layers = list()
    max_pooling_layers = list()
    conv_layers = list()

    for layer in layer_list:
        if "conv2d" in layer.name:
            conv_layers.append(layer)
        elif "max_pooling2d" in layer.name:
            max_pooling_layers.append(layer)
        elif "dense" in layer.name:
            dense_layers.append(layer)
        elif "dropout" in layer.name:
            dropout_layers.append(layer)

    if len(conv_layers):
        rez+=f"({conv_layers[0].kernel_size[0]},{conv_layers[0].kernel_size[1]})conv2d "
        for i in range(0,len(conv_layers)-1):
            rez+=f"{conv_layers[i].filters}->"
        rez+=f"{conv_layers[-1].filters}, "

    if len(max_pooling_layers):
        rez+=f"({max_pooling_layers[0].pool_size[0]},{max_pooling_layers[0].pool_size[1]})max pooling, "
    if len(dense_layers):
        rez += "dense "
        for i in range(0, len(dense_layers) - 1):
            rez += f"{dense_layers[i].units}->"
        rez += f"{dense_layers[-1].units}, "
    if len(dropout_layers):
        rez+= f"dropout {dropout_layers[0].rate}"
    return rez