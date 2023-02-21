import os

class Parameters():
	def __init__(self, args):
		self.n_processors = 8

		#Getting arguments from command line:

		Unity = args.unity
		KITTI = args.kitti
		Unreal = args.unreal

		
		# Path -  KITTI:
		# KITTI = False
		# Unity = True

		# KITTI=True
		# Unity=False

		# KITTI=False
		# Unity=False
		# Unreal = True

		if KITTI:
			self.data_dir =  '/home/mrblack/Projects_DL/DeepVO-pytorch/KITTI/'
			self.image_dir = self.data_dir + 'images/'
			self.pose_dir = self.data_dir + 'pose_GT/'

			self.train_video = ['00', '01', '02', '05', '08', '09']
			self.valid_video = ['04', '06', '07', '10']
			self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8


		# Path -  Unity:
		if Unity:
			self.data_dir =  '/home/mrblack/Projects_DL/DeepVO-pytorch/Unity/back_forward/'
			self.image_dir = self.data_dir + 'image_left/'
			self.pose_dir = self.data_dir + 'pose_left_kitti/'

			self.train_video = ['00']
			self.valid_video = ['00']
			self.partition=None

		if Unreal:
			self.data_dir =  '/home/mrblack/Projects_DL/DeepVO-pytorch/Unreal/test_image_7/'
			self.image_dir = self.data_dir + 'image_left/'
			self.pose_dir = self.data_dir + 'pose_left_kitti/'

			self.train_video = ['00']
			self.valid_video = ['00']
			self.partition=None

		# Data Preprocessing
		self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
		self.img_w = 608   # original size is about 1226, Original_Unity=1172
		self.img_h = 184   # original size is about 370, Original_Unity=530
		self.img_means =  (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
		self.img_stds =  (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
		self.minus_point_5 = True

		# Data info
		self.seq_len = (5, 7)
		self.sample_times = 3

		# Data info path
		# self.train_data_info_path = 'datainfo/train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
		# self.valid_data_info_path = 'datainfo/valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)

		self.train_data_info_path = 'datainfo/train_df_t000102050809_v04060710_pNone_seq5x7_sample3.pickle'
		self.valid_data_info_path = 'datainfo/valid_df_t000102050809_v04060710_pNone_seq5x7_sample3.pickle'

		# Model
		self.rnn_hidden_size = 1000
		self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
		self.rnn_dropout_out = 0.5
		self.rnn_dropout_between = 0   # 0: no dropout
		self.clip = None
		self.batch_norm = True
		# Training
		self.epochs = 10 #250
		self.batch_size = 6 #8
		self.pin_mem = True
		self.optim = {'opt': 'Adagrad', 'lr': 0.0005}
					# Choice:
					# {'opt': 'Adagrad', 'lr': 0.001}
					# {'opt': 'Adam'}
					# {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}
		
		# Pretrain, Resume training
		self.pretrained_flownet = None
								# Choice:
								# None
								# './pretrained/flownets_bn_EPE2.459.pth.tar'  
								# './pretrained/flownets_EPE1.951.pth.tar'
		self.resume = True  # resume training
		# self.resume = False
		self.resume_t_or_v = '.train'
		
		if KITTI:
			self.load_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
			self.load_optimizer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
		if Unity or Unreal:
			# self.load_model_path = 'models/t000102050809_v04060710_im184x608_s5x7_b8_rnn1000_optAdagrad_lr0.0005.model.train'
			# self.load_optimizer_path = 'models/t000102050809_v04060710_im184x608_s5x7_b8_rnn1000_optAdagrad_lr0.0005.optimizer.train'
			self.load_model_path = 'models/VO_pretrained_unity.model.train'
			self.load_optimizer_path = 'models/VO_pretrained_unity.optimizer.train'


		self.record_path = 'records/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))

		if KITTI:
			self.save_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
			self.save_optimizer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		
		if Unity or Unreal:
			self.save_model_path = 'models/VO_pretrained_unity.model.train'
			self.save_optimizer_path = 'models/VO_pretrained_unity.optimizer.train'
		
		if not os.path.isdir(os.path.dirname(self.record_path)):
			os.makedirs(os.path.dirname(self.record_path))
		if not os.path.isdir(os.path.dirname(self.save_model_path)):
			os.makedirs(os.path.dirname(self.save_model_path))
		if not os.path.isdir(os.path.dirname(self.save_optimizer_path)):
			os.makedirs(os.path.dirname(self.save_optimizer_path))
		if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
			os.makedirs(os.path.dirname(self.train_data_info_path))

# par = Parameters()