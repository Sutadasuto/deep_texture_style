import argparse
import IPython.display as display
import time

from distutils.util import strtobool

from image_utilities import *
from loss_utilities import *
from vgg_utilities import *


def main(args):
    if not os.path.exists("images"):
        os.mkdir("images")
    output_file_name = 'stylized-image_%s-steps.png'

    content_image = load_img(args.target_content, "content", args.max_dim)
    style_image = load_img(args.target_style, "style", args.max_dim)

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    with open(os.path.join("images", "used_parameters.txt"), "w+") as f:
        new_string = "Content layers:\n%s\nStyle layers:\n%s\n" % (str(content_layers), str(style_layers))
        for attribute in args.__dict__.keys():
            new_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
        f.write(new_string)

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    if args.initial_image_path:
        image = load_img(args.initial_image_path, target_type=None, max_dim=None, copy=False)
        image = tf.Variable(image)
        tensor_to_image(image).save(os.path.join("images", output_file_name % 0))
    else:
        if args.initial_image == "content":
            image = tf.Variable(content_image)
        elif args.initial_image == "style":
            image = tf.Variable(tf.image.resize(style_image, tf.shape(content_image)[1:-1]))
        else:
            from numpy.random import normal
            image = tf.convert_to_tensor(normal(0.5, 0.125, tf.shape(content_image)), dtype=tf.float32)
            image = tf.Variable(image)
            image.assign(clip_0_1(image))
            tensor_to_image(image).save(os.path.join("images", output_file_name % 0))

    opt = tf.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs, style_targets, num_style_layers, content_targets, num_content_layers,
                                      style_weight=1e-2, content_weight=1e4)
            loss += total_variation_loss(image, weight=args.variation_weight)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))
        return loss

    start = time.time()
    step = 0
    print("Epoch {} finished. Train step: {}/{}. Current time: {:.1f}s".format(0, step, args.steps,
                                                                               time.time() - start))
    for n in range(args.epochs):
        for m in range(args.steps):
            step += 1
            current_loss = train_step(image)
            print(".", end='')
        display.clear_output(wait=True)
        print("Epoch {} finished. Train step: {}/{}. Current time: {:.1f}s. Current loss: {:.2f}".format(n + 1, step, args.steps*args.epochs,
                                                                                   time.time() - start, current_loss.numpy()[0]))
        tensor_to_image(image).save(os.path.join("images", output_file_name % step))

    end = time.time()
    print("Total time: {:.1f}s".format(end - start))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("target_content", type=str, help="Path to image intended as content target.")
    parser.add_argument("target_style", type=str, help="Path to image intended as style target.")
    parser.add_argument("--initial_image", type=str, default="content", choices=["content", "style", "noise"],
                        help="One of 'content', 'style' or 'noise'. Ignored if 'initial_image_path' is not None")
    parser.add_argument("--initial_image_path", type=str, default=None,
                        help="Load this image as initial image for optimization. This assumes continuing a previous"
                             "training session.")
    parser.add_argument("--epochs", type=int, default=50, help="One epoch is completed after 'steps' iterations. "
                                                               "At epoch end a partial result image is saved in "
                                                               "'images'.")
    parser.add_argument("--steps", type=int, default=10, help="Number of iterations needed to finish a training epoch.")
    parser.add_argument("--style_weight", type=float, default=1e-2, help="Weight assigned to the style loss.")
    parser.add_argument("--content_weight", type=float, default=1e4, help="Weight assigned to the content loss.")
    parser.add_argument("--variation_weight", type=float, default=30.0, help="Weight assigned to the total variation loss.")
    parser.add_argument("--learning_rate", type=float, default=2e-2, help="Learning rate for Adam optimizer.")
    parser.add_argument("--max_dim", type=int, default=None, help="If provided, style and content image will be resized"
                                                                  "keeping their aspect ratio such that the longest "
                                                                  "dimension matches this argument as best as possible"
                                                                  "during training.")

    args_dict = parser.parse_args(args)
    for attribute in args_dict.__dict__.keys():
        if args_dict.__getattribute__(attribute) == "None":
            args_dict.__setattr__(attribute, None)
        if args_dict.__getattribute__(attribute) == "True" or args_dict.__getattribute__(attribute) == "False":
            args_dict.__setattr__(attribute, bool(strtobool(args_dict.__getattribute__(attribute))))
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)
