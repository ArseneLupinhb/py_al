#!/usr/bin/env python
# encoding: utf-8
'''
@author: al
@file: sentiment_classification_ernie.py
@time: 2020/6/5 1:01
@desc: 情感分类
'''
import paddlehub as hub

if __name__ == '__main__':
	module = hub.Module(name="ernie")
	dataset = hub.dataset.ChnSentiCorp()

	reader = hub.reader.ClassifyReader(
		dataset=dataset,
		vocab_path=module.get_vocab_path(),
		max_seq_len=128)

	strategy = hub.AdamWeightDecayStrategy(
		weight_decay=0.01,
		warmup_proportion=0.1,
		learning_rate=5e-5)

	config = hub.RunConfig(
		use_cuda=False,
		num_epoch=1,
		checkpoint_dir="ernie_txt_cls_turtorial_demo",
		batch_size=100,
		eval_interval=50,
		strategy=strategy)

	inputs, outputs, program = module.context(
		trainable=True, max_seq_len=128)

	# Use "pooled_output" for classification tasks on an entire sentence.
	pooled_output = outputs["pooled_output"]

	feed_list = [
		inputs["input_ids"].name,
		inputs["position_ids"].name,
		inputs["segment_ids"].name,
		inputs["input_mask"].name,
	]

	cls_task = hub.TextClassifierTask(
		data_reader=reader,
		feature=pooled_output,
		feed_list=feed_list,
		num_classes=dataset.num_labels,
		config=config)

	run_states = cls_task.finetune_and_eval()
