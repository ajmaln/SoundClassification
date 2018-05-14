self.TrainingDatasetMaker()
_, c = session.run([optimizer, cross_entropy], feed_dict={X: self.train_x, Y: train_y})