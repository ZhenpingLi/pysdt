import numpy as np

from algorithm.algorithm_def import AlgorithmDef
from algorithm.data_point import DataPoint
from training.training_set import TrainingSet


def get_lm_training_set(input_set: TrainingSet, algorithm: AlgorithmDef) -> TrainingSet:
    accumulator = 0
    sample_period_str = algorithm.get_attribute("sampleperiod") or "80"
    sample_period = float(sample_period_str)
    inputs = input_set.inputs[:, 0]
    outputs = input_set.outputs
    scale = sample_period / 10
    end_w = inputs[0] + sample_period
    prev_outputs = input_set.outputs[0]
    delta_list = []
    for i in range(len(input_set.outputs)):
        if inputs[i] >= end_w:
            delta = outputs[i]- prev_outputs
            if delta < 0:
                accumulator = 0
            else:
                accumulator = accumulator + delta/scale
            prev_outputs = outputs[i]
            delta_list.append(DataPoint(inputs[i], [accumulator]))
            end_w += sample_period
            accumulator = 0

    delta_inputs = np.array([dp.time for dp in delta_list]).reshape(-1, 1)
    delta_outputs = np.array([dp.data[0] for dp in delta_list])
    delta_dqf = np.ones(len(delta_outputs), dtype=np.int32)

    output_set = TrainingSet(mnemonic_id=input_set.mnemonic_id, inputs=delta_inputs, raw=delta_outputs, dqf=delta_dqf, outputs=delta_outputs)
    return output_set