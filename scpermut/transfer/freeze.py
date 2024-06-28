def freeze_layers(ae, layers_to_freeze):
    """
    Freezes specified layers in the model.

    ae: Model to freeze layers in.
    layers_to_freeze: List of layers to freeze.
    """
    for layer in layers_to_freeze:
        layer.trainable = False


def freeze_block(ae, strategy):
    if strategy == "all_but_classifier_branch":
        layers_to_freeze = [
            ae.dann_discriminator,
            ae.enc,
            ae.dec,
            ae.ae_output_layer,
        ]
    elif strategy == "all_but_classifier":
        layers_to_freeze = [ae.dann_discriminator, ae.dec, ae.ae_output_layer]
    elif strategy == "all_but_dann_branch":
        layers_to_freeze = [ae.classifier, ae.dec, ae.ae_output_layer, ae.enc]
    elif strategy == "all_but_dann":
        layers_to_freeze = [ae.classifier, ae.dec, ae.ae_output_layer]
    elif strategy == "all_but_autoencoder":
        layers_to_freeze = [ae.classifier, ae.dann_discriminator]
    else:
        raise ValueError("Unknown freeze strategy: " + strategy)

    self.freeze_layers(ae, layers_to_freeze)


def freeze_all(ae):
    for l in ae.layers:
        l.trainable = False


def unfreeze_all(ae):
    for l in ae.layers:
        l.trainable = True
