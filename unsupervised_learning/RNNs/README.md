# RNNs
This project focuses on Recurrent Neural Networks (RNNs), providing an overview of their types and functionalities. RNNs are a category of neural networks designed to analyze sequential data, such as text, speech, and time series. They are known for their ability to maintain a hidden state that evolves with new inputs, making them suitable for recognizing patterns in sequences. 

All files utilize Numpy only, with a focus on understanding the processes behind different types of RNNs.

## Recurrent Neural Networks
RNNs are neural networks that can process sequential data by maintaining a hidden state that updates with each new input. This feature makes RNNs ideal for tasks involving sequential data, such as natural language processing and time series analysis.

## Gated Recurrent Units
GRUs are a variant of RNNs that incorporate gating mechanisms to control the flow of information within the network. By simplifying the architecture compared to LSTMs, GRUs offer computational efficiency while still being able to capture long-term dependencies in data.

## Long Short-Term Memory
LSTMs are another variant of RNNs that introduce a memory cell to store information over long periods. This design addresses the vanishing gradient problem, allowing LSTMs to learn and remember over long sequences, which is essential for various applications.

## Deep RNNs
Deep RNNs consist of multiple layers of RNNs stacked on top of each other. This layered structure enables the network to learn more complex representations of the input data, enhancing its capability to perform tasks that require understanding intricate patterns in sequential data.

## Bidirectional RNNs
Bidirectional RNNs process the input data in both forward and backward directions. This dual-directional processing allows the network to utilize both past and future context simultaneously, improving its predictive power and accuracy in tasks such as sentiment analysis and time series forecasting.
