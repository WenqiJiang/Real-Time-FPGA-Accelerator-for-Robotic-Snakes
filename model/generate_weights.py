import numpy as np

if __name__ == "__main__":

####################        CONSTANTS, DONT CHANGE          ####################

    LSTM_INPUT_SIZE_1 = 14
    LSTM_STATE_SIZE_1 = 16

    LSTM_INPUT_SIZE_2 = LSTM_STATE_SIZE_1
    LSTM_STATE_SIZE_2 = 16

    FC_INPUT_SIZE = LSTM_STATE_SIZE_2
    FC_OUTPUT_SIZE = 10

    COMPUTE_TIME = 1000000

    print ("INFO: Start generating weights")

    # Gaussian Distribution
    # LSTM Layer 1 weights
    np.random.seed(123)
    forget_gate_kernel_last_state_1 = np.random.randn(
        LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1)
    np.savetxt("forget_gate_kernel_last_state_1.txt",
            forget_gate_kernel_last_state_1)

    np.random.seed(124)
    forget_gate_kernel_input_state_1 = np.random.randn(
        LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1)
    np.savetxt("forget_gate_kernel_input_state_1.txt",
            forget_gate_kernel_input_state_1)

    np.random.seed(125)
    forget_gate_bias_1 = np.random.randn(LSTM_STATE_SIZE_1)
    np.savetxt("forget_gate_bias_1.txt", forget_gate_bias_1)

    np.random.seed(126)
    input_gate_kernel_last_state_1 = np.random.randn(
        LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1)
    np.savetxt("input_gate_kernel_last_state_1.txt",
            input_gate_kernel_last_state_1)

    np.random.seed(127)
    input_gate_kernel_input_state_1 = np.random.randn(
        LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1)
    np.savetxt("input_gate_kernel_input_state_1.txt",
            input_gate_kernel_input_state_1)

    np.random.seed(128)
    input_gate_bias_1 = np.random.randn(LSTM_STATE_SIZE_1)
    np.savetxt("input_gate_bias_1.txt", input_gate_bias_1)

    np.random.seed(129)
    tanh_gate_kernel_last_state_1 = np.random.randn(
        LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1)
    np.savetxt("tanh_gate_kernel_last_state_1.txt",
            tanh_gate_kernel_last_state_1)

    np.random.seed(130)
    tanh_gate_kernel_input_state_1 = np.random.randn(
        LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1)
    np.savetxt("tanh_gate_kernel_input_state_1.txt",
            tanh_gate_kernel_input_state_1)

    np.random.seed(131)
    tanh_gate_bias_1 = np.random.randn(LSTM_STATE_SIZE_1)
    np.savetxt("tanh_gate_bias_1.txt", tanh_gate_bias_1)

    np.random.seed(132)
    output_gate_kernel_last_state_1 = np.random.randn(
        LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1)
    np.savetxt("output_gate_kernel_last_state_1.txt",
            output_gate_kernel_last_state_1)

    np.random.seed(133)
    output_gate_kernel_input_state_1 = np.random.randn(
        LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1)
    np.savetxt("output_gate_kernel_input_state_1.txt",
            output_gate_kernel_input_state_1)

    np.random.seed(134)
    output_gate_bias_1 = np.random.randn(LSTM_STATE_SIZE_1)
    np.savetxt("output_gate_bias_1.txt", output_gate_bias_1)

    # LSTM Layer 2 weights
    np.random.seed(135)
    forget_gate_kernel_last_state_2 = np.random.randn(
        LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2)
    np.savetxt("forget_gate_kernel_last_state_2.txt",
            forget_gate_kernel_last_state_2)

    np.random.seed(136)
    forget_gate_kernel_input_state_2 = np.random.randn(
        LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2)
    np.savetxt("forget_gate_kernel_input_state_2.txt",
            forget_gate_kernel_input_state_2)

    np.random.seed(137)
    forget_gate_bias_2 = np.random.randn(LSTM_STATE_SIZE_2)
    np.savetxt("forget_gate_bias_2.txt", forget_gate_bias_2)

    np.random.seed(138)
    input_gate_kernel_last_state_2 = np.random.randn(
        LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2)
    np.savetxt("input_gate_kernel_last_state_2.txt",
            input_gate_kernel_last_state_2)

    np.random.seed(139)
    input_gate_kernel_input_state_2 = np.random.randn(
        LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2)
    np.savetxt("input_gate_kernel_input_state_2.txt",
            input_gate_kernel_input_state_2)

    np.random.seed(140)
    input_gate_bias_2 = np.random.randn(LSTM_STATE_SIZE_2)
    np.savetxt("input_gate_bias_2.txt", input_gate_bias_2)

    np.random.seed(141)
    tanh_gate_kernel_last_state_2 = np.random.randn(
        LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2)
    np.savetxt("tanh_gate_kernel_last_state_2.txt",
            tanh_gate_kernel_last_state_2)

    np.random.seed(142)
    tanh_gate_kernel_input_state_2 = np.random.randn(
        LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2)
    np.savetxt("tanh_gate_kernel_input_state_2.txt",
            tanh_gate_kernel_input_state_2)

    np.random.seed(143)
    tanh_gate_bias_2 = np.random.randn(LSTM_STATE_SIZE_2)
    np.savetxt("tanh_gate_bias_2.txt", tanh_gate_bias_2)

    np.random.seed(144)
    output_gate_kernel_last_state_2 = np.random.randn(
        LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2)
    np.savetxt("output_gate_kernel_last_state_2.txt",
            output_gate_kernel_last_state_2)

    np.random.seed(145)
    output_gate_kernel_input_state_2 = np.random.randn(
        LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2)
    np.savetxt("output_gate_kernel_input_state_2.txt",
            output_gate_kernel_input_state_2)

    np.random.seed(146)
    output_gate_bias_2 = np.random.randn(LSTM_STATE_SIZE_2)
    np.savetxt("output_gate_bias_2.txt", output_gate_bias_2)

    # Layer 1, a super large array, contain the inputs of all steps
    np.random.seed(147)
    lstm_input_state_1 = np.random.randn(LSTM_INPUT_SIZE_1 * COMPUTE_TIME)
    np.savetxt("lstm_input_state_1.txt", lstm_input_state_1)

    # fc weights
    np.random.seed(148)
    fc_kernel = np.random.randn(FC_OUTPUT_SIZE * FC_INPUT_SIZE)
    np.savetxt("fc_kernel.txt", fc_kernel)

    np.random.seed(149)
    fc_bias = np.random.randn(FC_OUTPUT_SIZE)
    np.savetxt("fc_bias.txt", fc_bias)
    print ("INFO: Finish generating weights")
