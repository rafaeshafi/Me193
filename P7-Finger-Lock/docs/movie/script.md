# Voice-over script

*642 frames at 3 a second = 214 s. Play `movie_silent.mp4` and read each paragraph while its scene is on screen (QuickTime > File > New Screen Recording with the microphone on, or Zoom with 'record'). The times are where each scene starts.*

## 1. Title  (0:00 to 0:06)

This is how my finger lock sees. A camera looks at a square on the desk, a small neural network counts the fingers in it, and a lock policy decides whether the safe opens. I built the network myself in PyTorch, trained it on my own hand, and then moved it onto the Arduino UNO Q. Let me walk through what happens to one picture.

## 2. From a picture to numbers  (0:06 to 0:19)

The webcam gives a colour picture. I only keep the square where the hand goes, turn it to grey and shrink it to 64 by 64. To the network a picture is nothing but a grid of 4,096 numbers, one per pixel, 0 for black and 1 for white. Everything that follows is arithmetic on these numbers.

## 3. Convolution  (0:19 to 0:46)

The first layer is a convolution. A tiny 3 by 3 window of weights, called a kernel or filter, slides over the picture one pixel at a time. At every position it multiplies the nine pixels under it by its nine weights and adds them up; that single number goes into a new picture called a feature map. The weights were learned, not chosen by me: this filter happens to respond to edges running one way. The first layer has sixteen different filters, so it makes sixteen feature maps, each looking for something different.

## 4. ReLU  (0:46 to 0:58)

After each convolution comes ReLU, the rectified linear unit. It is the simplest rule there is: if a number is negative, make it zero; otherwise leave it alone. Without this bend the whole network would collapse into one big multiplication and could only draw straight lines. The bend is what lets it learn shapes.

## 5. Max pooling  (0:58 to 1:13)

Then max pooling. The feature map is cut into 2 by 2 blocks and only the biggest number in each block survives. The map shrinks to half its width and half its height. We lose exact positions, which is the point: a finger is a finger whether it sits two pixels left or right, and the next layer gets four times less to look at.

## 6. Three layers deep  (1:13 to 1:29)

My network does this three times. Convolution, ReLU, pooling, with sixteen filters, then thirty-two, then sixty-four. Each stage sees a bigger piece of the hand: the first finds edges, the second joins edges into strokes, the third tells a finger-shaped stroke from a palm. By the end the picture is sixty-four tiny 8 by 8 maps. That whole stack has twenty-three thousand weights; the picture went in as four thousand numbers and comes out as four thousand numbers that mean something.

## 7. Flatten, linear layer, softmax  (1:29 to 1:46)

Now the decision. The 64 little maps are unrolled into one long line of 4,096 numbers. A linear layer takes a weighted sum of all of them for each of 64 hidden units, then one more linear layer turns those into six scores, one per answer: zero to five fingers. Softmax squashes the six scores into probabilities that add up to one. The biggest probability is the network's answer. For this hand it says three fingers, and you can read how sure it is.

## 8. Cross-entropy loss  (1:46 to 1:59)

How does it learn? During training the right answer is known, because I typed it when I took the picture. The loss is cross entropy: minus the log of the probability the network gave the true answer. If it was sure and right, the loss is near zero. If it gave the true answer only 10 percent, the loss is 2.3. One number says how wrong the whole network was on that picture.

## 9. Gradient descent  (1:59 to 2:32)

Every weight gets nudged to make that loss smaller. Picture the loss as a bowl and the weights as a ball somewhere on its wall. The gradient is the slope under the ball. Gradient descent takes a small step downhill: new weight equals old weight minus the learning rate times the slope. Do that for all 286 thousand weights, on every batch of pictures, for forty passes over the data, and the loss curve falls like this. The real curve from my training run is drawn here, one epoch per step.

## 10. Back-propagation  (2:32 to 2:53)

Where do the slopes come from? Back-propagation. The loss is computed at the end, then the chain rule walks backwards through the layers: how much did the last layer's weights change the loss, then the layer before, all the way back to the first filter. Each layer only needs the gradient handed to it from the layer after it. PyTorch does the bookkeeping; the one line in my code is loss dot backward.

## 11. More pictures for free  (2:53 to 3:08)

I took around a thousand pictures, but the network wants more. So every time a picture is used, it is rotated a little, shifted, scaled, flipped and re-lit, differently each time. The label stays the same: a rotated three is still a three. This is augmentation, and it is what makes the lock forgive a hand that is not exactly where the training hand was.

## 12. The lock policy  (3:08 to 3:29)

Finally the lock. Press the button and it listens. Ten times a second the network says what it sees. A digit counts only when the same answer comes back for eight frames in a row with high confidence, so a flicker is ignored; then the hand has to leave before the next digit starts. Three digits make a code. If they match, the servo opens the latch for five seconds. If not, red light, and after three wrong tries the safe ignores the button for half a minute.

## 13. The end  (3:29 to 3:34)

That is the whole network: numbers in, convolutions that look for shapes, pooling that forgives position, a linear layer that votes, and a loss that teaches it all through gradient descent. The same weights now run on the UNO Q in plain numpy, and the safe opens for my hand.
