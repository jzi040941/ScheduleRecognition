import keras

class Concurrence(Layer):
	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]
		self.input_dim = input_shape[2]

		self.W = self.add_weight((self.input_dim, 1),
								  initializer=self.init,
								  name='{}_W'.format(self.name),
								  regularizer=self.W_regularizer)
		super(Concurrence, self).build(input_shape)
	
	def call(self, x, mask=None):
		attention = K.softmax(K.squeeze(K.dot(x, self.W), 2))
		return K.batch_dot(x, attention, (1, 1))

model = Sequential()
model.add(Bidirectional(GRU(hidden_size, return_sequences=True), merge_mode='concat',
                            input_shape=(None, input_size)))

model.add(Concurrence())
model.add(RepeatVector(max_out_seq_len + 1))
model.add(GRU(hidden_size * 2, return_sequences=True))
model.add(TimeDistributed(Dense(output_dim=output_size, activation="softmax")))
model.compile(loss="categorical_crossentropy", optimizer="rms_prop")