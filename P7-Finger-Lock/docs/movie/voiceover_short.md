# Finger Lock voice-over (read-aloud version)

Movie: 3:54. Each part is short enough to say in the time its scene is on screen.

**0:00 Title**
This is how my finger lock sees a hand. Let me walk you through one picture.

**0:06 Picture to numbers**
The webcam looks at a square on my desk. I crop to that square, make it grey, and shrink it to 128 by 128. To the net, that's just sixteen thousand numbers.

**0:19 Convolution**
First is a convolution. A little 3 by 3 window of weights slides across the picture. At every spot it multiplies the nine pixels under it by its nine weights and adds them up. That one number goes into a new picture called a feature map. I didn't pick those weights, the net learned them. This one happens to like edges. The first layer has sixteen of these filters, so it makes sixteen maps.

**0:46 ReLU**
Next is ReLU. If a number is negative, it becomes zero. Otherwise it stays. That's the whole rule, and without it the net could only ever draw straight lines.

**0:58 Max pooling**
Then max pooling. I cut the map into 2 by 2 blocks and keep only the biggest number in each. The map is now half the size, and it stops caring if a finger moved a pixel or two.

**1:13 Three layers**
The net does that three times, with sixteen, then thirty-two, then sixty-four filters. Early layers find edges. Later ones find finger shapes. At the end I have sixty-four small 16 by 16 maps.

**1:29 Linear layers and softmax**
Now the decision. I unroll those maps into one long row of numbers. Two linear layers boil it down to six scores, one for each answer, zero to five fingers. Softmax turns the scores into probabilities. The biggest one wins.

**1:46 Loss**
So how does it learn? In training I know the right answer. The loss is one number that says how wrong the net was. Sure and right, it's near zero. Sure and wrong, it's big.

**1:59 Gradient descent**
Training is just making that number smaller. Think of the loss as a bowl, and the weights as a ball sitting on the side of it. The gradient is the slope under the ball. So I take a small step downhill. New weight equals old weight, minus the learning rate times the slope.

It's the same thing I did by hand in the spreadsheet with m and b. The difference is that this net has about a million weights instead of two.

I do that for every weight, on every batch of pictures, a hundred times through the data. And this is my real loss curve from training. You can see it drop fast at the start and then flatten out.

**2:52 Backprop**
Where do the slopes come from? Backpropagation. The loss is at the end, so the chain rule works backwards, layer by layer, all the way to the first filter. Each layer only needs what the layer after it hands back. In PyTorch that's one line: loss dot backward.

**3:13 Augmentation**
I only took about nine hundred pictures. So every time the net sees one, I rotate it, shift it, scale it and change the lighting a bit. A rotated three is still a three.

**3:28 Lock policy**
Last part, the lock. Press the button and it starts listening. A digit only counts if the net says the same number for three seconds straight. Then my hand has to leave before the next one. Three digits make the code. Right code, the bolt opens for ten seconds. Three wrong tries, it locks me out.

**3:49 End**
And that's it. The same weights run on the UNO Q, and the safe opens for my hand.
