import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
from typing import Any
import h5py
import copy

import numpy as np
import pandas as pd
import torch
# import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
import string

translator_rm_punct = str.maketrans('', '', string.punctuation)
def rm_punct(s):
    new_string = s.translate(translator_rm_punct)
    return new_string

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class video_sample(object):
    def __init__(self, num_frame) -> None:
        self.num_frame = num_frame
    
    def __call__(self, video: Any) -> Any:
        frames_num = len(video)
        if frames_num < self.num_frame:
            zero_pad = np.zeros((self.num_frame - frames_num, video.shape[-1]), dtype=video.dtype)
            video = np.concatenate((video, zero_pad), axis=0)
            frames_num = len(video)
        frames_per_segment = frames_num // self.num_frame
        frames_remain = frames_num % self.num_frame
        random_samples = []
        sum_length = 0
        for i in range(self.num_frame):
            if i < frames_remain:
                random_samples.append(sum_length + random.randint(0, frames_per_segment))
                sum_length += (frames_per_segment + 1)
            else:
                random_samples.append(sum_length + random.randint(0, frames_per_segment - 1))
                sum_length += frames_per_segment
        
        res = None
        for i in range(self.num_frame):
            current = video[random_samples[i]][np.newaxis, :]
            if res is None:
                res = current
            else:
                res = np.concatenate((res, current), axis=0)
        return res

# class caption_tokenizer(object):
#     def __init__(self, tokenizer, eos_id) -> None:
#         self.tokenizer = tokenizer
#         self.eos_id = eos_id
    
#     def __call__(self, caption: Any) -> Any:
#         tokenizer_results = self.tokenizer.encode(caption)
#         return tokenizer_results.append(self.eos_id)

class My_collate_fn(object):
    def __init__(self, pad_token, eos_token) -> None:
        self.pad_token = pad_token
        self.eos_token = eos_token
    
    def __call__(self, batch: Any) -> Any:
        res_video = None
        res_text = None
        input_lens_list = [len(w) for _, w in batch]
        max_input_len = max(input_lens_list)
        for btc_idx in range(len(batch)):
            video, caption = batch[btc_idx]
            video = torch.tensor(video).unsqueeze(0)
            if res_video is None:
                res_video = video
            else:
                res_video = torch.cat((res_video, video), dim=0)
            input_len = len(caption)
            caption.extend([self.eos_token])
            caption.extend([self.pad_token] * (max_input_len - input_len))
            caption = torch.tensor(caption, dtype=torch.long).unsqueeze(0)
            if res_text is None:
                res_text = caption
            else:
                res_text = torch.cat((res_text, caption), dim=0)
        return {'video': res_video, 'text': res_text}

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
    num_shards = len(shards_list)
    return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('text' in sample)
    has_image = ('image' in sample)
    if sample['image'] is None or sample['text'] is None:
        return False
    if len(sample['text']) > 51:
        return False
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, model=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
        
    my_collator = My_collate_fn(model.pad_token_id, model.eos_ad_token_id)
    video_processer = video_sample(model.video_length)
    # caption_processer = caption_tokenizer(tokenizer, 170031)

    pipeline.extend([
        # wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="npy", text="txt"),
        wds.map_dict(image=video_processer, text=lambda text: model.tokenizer.encode(text)),
        wds.select(filter_no_caption_or_no_image), # 在这里把tokenizer之后的结果进行筛选丢弃
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train, collation_fn=my_collator)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        # collate_fn=my_collator,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
  
class JsonDataset(Dataset):
    def __init__(self, input_filename, movie_feature_path, char_feature_path, model, if_special_prompt, if_train_drop, char_prompt_type, previous_video_num, if_video_split):
        logging.debug(f'Loading json data from {input_filename}.')
        with open(input_filename) as f:
            self.data = json.load(f)
        self.video = h5py.File(movie_feature_path,'r')
        self.char_feature = torch.load(char_feature_path)
        self.tokenizer = model.tokenizer
        self.model = model
        self.video_processer = video_sample(self.model.video_length)
        self.fps = 5
        self.if_special_prompt = if_special_prompt
        self.if_train_drop = if_train_drop
        self.char_prompt_type = char_prompt_type
        self.previous_video_num = previous_video_num
        self.if_video_split = if_video_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start = round(float(self.data[idx]['start'])) * self.fps
        end = round(float(self.data[idx]['end'])) * self.fps
        if start < 0 or end < 0:
            return [], None, None, None
        if start == end:
            start = math.floor(float(self.data[idx]['start'])) * self.fps
            end = math.ceil(float(self.data[idx]['end'])) * self.fps
            
        movie_id = self.data[idx]['movie_id']
        ad = self.data[idx]['ad']
        previous_ad = self.data[idx]['context'][0]['ad']

        char = self.data[idx]['ad_chars_in_chars']
        movie = self.video[movie_id][:][start:end+1]
        if len(movie) == 0:
            print('?', self.data[idx]['start'])
            return [], None, None, None
        video = self.video_processer(movie)

        if self.previous_video_num > 0: # assume as 1 now
            previous_start = round(float(self.data[idx]['context'][0]['start'])) * self.fps
            previous_end = round(float(self.data[idx]['context'][0]['end'])) * self.fps
            if previous_start < 0 or previous_end < 0:
                return [], None, None, None
            if previous_start == previous_end:
                previous_start = math.floor(float(self.data[idx]['context'][0]['start'])) * self.fps
                previous_end = math.ceil(float(self.data[idx]['context'][0]['end'])) * self.fps

            previous_movie_id = self.data[idx]['movie_id']
            previous_movie = self.video[previous_movie_id][:][previous_start:previous_end+1]
            previous_video = self.video_processer(previous_movie)
            video = np.concatenate((previous_video, video), axis=0)

        text = self.tokenizer.encode(ad)[1:]
        previous_text = self.tokenizer.encode(previous_ad)[1:]
        if len(text) > 51 or len(previous_text) > 51:
            return [], None, None, None
        char_text = None
        char_image = None
        char_prompt = ' played by '
        for i, c in enumerate(char):
            id, name, role = c.values()
            if self.if_train_drop == 1:
                if role.split(' ')[0].endswith('.'):  # likely a prefix
                    try:
                        role = rm_punct(role.split(' ')[1])
                    except:
                        role = role.split(' ')[0]  # maybe an initial
                else:
                    role = role.split(' ')[0]
            
            if self.char_prompt_type <= 1:
                char_image_fea = self.char_feature[movie_id+'_'+id]
                if self.char_prompt_type == 0:
                    char_text_ = role + char_prompt + name
                else:
                    assert self.char_prompt_type == 1
                    char_text_ = role
                char_text_embed = self.tokenizer.encode(char_text_)[1:]
                for n in range(self.model.char_length):
                    char_text_embed.append(self.model.image_token_id)
                # char_text_embed.append(self.model.image_token_id)
                if i == len(char) - 1:
                    char_text_embed.extend(self.tokenizer.encode('.')[1:])
                else:
                    char_text_embed.extend(self.tokenizer.encode(';')[1:])
                
                if char_text is None:
                    char_text = char_text_embed
                else:
                    char_text.extend(char_text_embed)
                if char_image is None:
                    char_image = char_image_fea
                else:
                    char_image = np.concatenate((char_image, char_image_fea), axis=0)
            else:
                if self.char_prompt_type == 2:
                    char_text_ = role + char_prompt + name
                else:
                    assert self.char_prompt_type == 3
                    char_text_ = role
                char_text_embed = self.tokenizer.encode(char_text_)[1:]
                if i == len(char) - 1:
                    char_text_embed.extend(self.tokenizer.encode('.')[1:])
                else:
                    char_text_embed.extend(self.tokenizer.encode(';')[1:])
                
                if char_text is None:
                    char_text = char_text_embed
                else:
                    char_text.extend(char_text_embed)
                
        full_text = []
        if char_text is not None:
            if self.if_special_prompt == 1:
                full_text.append(self.model.img_start_token_id)
            full_text.extend(self.tokenizer.encode('Possible characters:')[1:])
            full_text.extend(char_text)
            if self.if_special_prompt == 1:
                full_text.append(self.model.img_end_token_id)
        full_text.extend(self.tokenizer.encode('Describe ')[1:])
        if self.if_special_prompt == 1:
            full_text.append(self.model.video_start_token_id)
        if self.if_video_split == 0:
            for i in range(self.model.video_token_length):
                full_text.append(self.model.video_token_id)
        else:
            for i in range(self.model.video_token_length * (1 + self.previous_video_num)):
                full_text.append(self.model.video_token_id)
        if self.if_special_prompt == 1:
            full_text.append(self.model.video_end_token_id)
        full_text.extend(self.tokenizer.encode(':')[1:])
        if self.if_special_prompt == 1:
            full_text.append(self.model.bos_ad_token_id)

        ad_start_ids = len(full_text) - 1

        full_text_neg = copy.deepcopy(full_text)
        full_text_neg.extend(previous_text)
        full_text_neg.append(self.model.eos_ad_token_id)

        full_text.extend(text)
        full_text.append(self.model.eos_ad_token_id)
        
        return full_text, full_text_neg, char_image, video, ad_start_ids
        
   
class JsonDataset_EVAL(Dataset):
    def __init__(self, eval_data, movie_feature_path, char_feature_path, model, movie_id, if_special_prompt, if_train_drop, char_prompt_type, previous_video_num, if_video_split):
        # logging.debug(f'Loading json data from {input_filename}.')
        self.data = eval_data
        self.video = h5py.File(movie_feature_path,'r')
        self.char_feature = torch.load(char_feature_path)
        self.tokenizer = model.tokenizer
        self.model = model
        self.fps = 5
        self.movie_id = movie_id
        self.if_special_prompt = if_special_prompt
        self.if_train_drop = if_train_drop
        self.char_prompt_type = char_prompt_type
        self.previous_video_num = previous_video_num
        self.if_video_split = if_video_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start = round(float(self.data[idx]['start'])) * self.fps
        end = round(float(self.data[idx]['end'])) * self.fps
        if start < 0 or end < 0:
            return [], None, None, None
        if start == end:
            start = math.floor(float(self.data[idx]['start'])) * self.fps
            end = math.ceil(float(self.data[idx]['end'])) * self.fps
            # print(self.data[idx]['start'])
        movie_id = self.movie_id
        ad = self.data[idx]['ad']
        
        char = self.data[idx]['prob']
        
        movie = self.video[movie_id][:][start:end+1]
        if len(movie) == 0:
            print('?', self.data[idx]['start'])
            return [], None, None, None
        ids = np.linspace(0, len(movie) - 1, num=self.model.video_length, dtype=int)
        video = movie[ids]

        if self.previous_video_num > 0: # assume as 1 now
            previous_start = round(float(self.data[idx]['context'][0]['start'])) * self.fps
            previous_end = round(float(self.data[idx]['context'][0]['end'])) * self.fps
            if previous_start < 0 or previous_end < 0:
                return [], None, None, None
            if previous_start == previous_end:
                previous_start = math.floor(float(self.data[idx]['context'][0]['start'])) * self.fps
                previous_end = math.ceil(float(self.data[idx]['context'][0]['end'])) * self.fps

            previous_movie_id = self.movie_id
            previous_movie = self.video[previous_movie_id][:][previous_start:previous_end+1]
            previous_ids = np.linspace(0, len(previous_movie) - 1, num=self.model.video_length, dtype=int)
            previous_video = previous_movie[previous_ids]
            video = np.concatenate((previous_video, video), axis=0)

        char_text = None
        char_image = None
        char_prompt = ' played by '
        for i, c in enumerate(char):
            id, name, role = c.values()
            if self.if_train_drop == 1:
                if role.split(' ')[0].endswith('.'):  # likely a prefix
                    try:
                        role = rm_punct(role.split(' ')[1])
                    except:
                        role = role.split(' ')[0]  # maybe an initial
                else:
                    role = role.split(' ')[0]
            
            if self.char_prompt_type <= 1:
                char_image_fea = self.char_feature[movie_id+'_'+id]
                if self.char_prompt_type == 0:
                    char_text_ = role + char_prompt + name
                else:
                    assert self.char_prompt_type == 1
                    char_text_ = role
                char_text_embed = self.tokenizer.encode(char_text_)[1:]
                for n in range(self.model.char_length):
                    char_text_embed.append(self.model.image_token_id)
                
                if i == len(char) - 1:
                    char_text_embed.extend(self.tokenizer.encode('.')[1:])
                else:
                    char_text_embed.extend(self.tokenizer.encode(';')[1:])
                
                if char_text is None:
                    char_text = char_text_embed
                else:
                    char_text.extend(char_text_embed)
                if char_image is None:
                    char_image = char_image_fea
                else:
                    char_image = np.concatenate((char_image, char_image_fea), axis=0)
            else:
                if self.char_prompt_type == 2:
                    char_text_ = role + char_prompt + name
                else:
                    assert self.char_prompt_type == 3
                    char_text_ = role
                char_text_embed = self.tokenizer.encode(char_text_)[1:]
                if i == len(char) - 1:
                    char_text_embed.extend(self.tokenizer.encode('.')[1:])
                else:
                    char_text_embed.extend(self.tokenizer.encode(';')[1:])
                
                if char_text is None:
                    char_text = char_text_embed
                else:
                    char_text.extend(char_text_embed)
        full_text = []
        if char_text is not None:
            if self.if_special_prompt == 1:
                full_text.append(self.model.img_start_token_id)
            full_text.extend(self.tokenizer.encode('Possible characters:')[1:])
            full_text.extend(char_text)
            if self.if_special_prompt == 1:
                full_text.append(self.model.img_end_token_id)
        full_text.extend(self.tokenizer.encode('Describe ')[1:])
        if self.if_special_prompt == 1:
            full_text.append(self.model.video_start_token_id)
        if self.if_video_split == 0:
            for i in range(self.model.video_token_length):
                full_text.append(self.model.video_token_id)
        else:
            for i in range(self.model.video_token_length * (1 + self.previous_video_num)):
                full_text.append(self.model.video_token_id)
        if self.if_special_prompt == 1:
            full_text.append(self.model.video_end_token_id)
        full_text.extend(self.tokenizer.encode(':')[1:])
        if self.if_special_prompt == 1:
            full_text.append(self.model.bos_ad_token_id)
        
        return full_text, char_image, video, [ad]      
    
class My_collate_fn_json(object):
    def __init__(self, pad_token) -> None:
        self.pad_token = pad_token
    
    def __call__(self, batch: Any) -> Any:
        res_video = None
        res_text = None
        res_text_neg = None
        res_image = None
        res_image_idx = []
        res_video_idx = {}
        ad_start_ids_batch = []
        input_lens_list = [len(w) for w, *_ in batch]
        max_input_len = max(input_lens_list)
        real_idx = 0
        char_idx = 0

        input_lens_list_neg = [0]
        for _, w, *_ in batch:
            if w is not None:
                input_lens_list_neg.append(len(w))
        max_input_len_neg = max(input_lens_list_neg)

        for btc_idx in range(len(batch)):
            caption = batch[btc_idx][0]
            if len(caption) == 0:
                continue

            caption_neg = batch[btc_idx][1]
            char_image = batch[btc_idx][2]
            video = batch[btc_idx][3]
            ad_start_ids = batch[btc_idx][4]
            
            ad_start_ids_batch.append(ad_start_ids)

            video = torch.tensor(video).unsqueeze(0)
            image = None
            res_video_idx[real_idx] = []
            if char_image is not None:
                image = torch.tensor(char_image)
                for l in range(len(image)):
                    res_image_idx.append(real_idx)
                    res_video_idx[real_idx].append(char_idx)
                    char_idx += 1

            if res_video is None:
                res_video = video
            else:
                res_video = torch.cat((res_video, video), dim=0)
            
            if res_image is None:
                res_image = image
            elif image is not None:
                res_image = torch.cat((res_image, image), dim=0)

            input_len = len(caption)
            caption.extend([self.pad_token] * (max_input_len - input_len))
            
            input_len_neg = len(caption_neg)
            caption_neg.extend([self.pad_token] * (max_input_len_neg - input_len_neg))
            
            caption = torch.tensor(caption, dtype=torch.long).unsqueeze(0)
            
            caption_neg = torch.tensor(caption_neg, dtype=torch.long).unsqueeze(0)
            # except Exception as e:
            #     print(caption)
            if res_text is None:
                res_text = caption
            else:
                res_text = torch.cat((res_text, caption), dim=0)
            real_idx += 1

            if res_text_neg is None:
                res_text_neg = caption_neg
            else:
                res_text_neg = torch.cat((res_text_neg, caption_neg), dim=0)

        if len(res_image_idx) == 0:
            res_image_idx = None
        
        return {'video': res_video, 'text': res_text, 'text_neg': res_text_neg, 'char_image': res_image, 'image_idx': res_image_idx, 'video_idx': res_video_idx, 'ad_start_ids': ad_start_ids_batch}
        
        
class My_collate_fn_json_eval(object):
    def __init__(self, pad_token) -> None:
        self.pad_token = pad_token
    
    def __call__(self, batch: Any) -> Any:
        res_video = None
        res_prompt = None
        res_image = None
        res_ad = None
        res_image_idx = []
        res_video_idx = {}
        input_lens_list = [len(w) for w, *_ in batch]
        max_input_len = max(input_lens_list)
        attention_mask = torch.ones(len(batch), max_input_len)
        real_idx = 0
        char_idx = 0
        for btc_idx in range(len(batch)):
            prompt = batch[btc_idx][0]
            char_image = batch[btc_idx][1]
            video = batch[btc_idx][2]
            ad = batch[btc_idx][3]  # 
            
            if len(prompt) == 0 and video is None and ad is None:   # and char_image is None 
                continue
            video = torch.tensor(video).unsqueeze(0)
            image = None
            res_video_idx[real_idx] = []
            if char_image is not None:
                image = torch.tensor(char_image)
                for l in range(len(image)):
                    res_image_idx.append(real_idx)
                    res_video_idx[real_idx].append(char_idx)
                    char_idx += 1
            if res_video is None:
                res_video = video
            else:
                res_video = torch.cat((res_video, video), dim=0)
            
            if res_image is None:
                res_image = image
            elif image is not None:
                res_image = torch.cat((res_image, image), dim=0)

            if res_ad is None:
                res_ad = ad
            else:
                res_ad.extend(ad)

            input_len = len(prompt)
            num_pad = (max_input_len - input_len)
            full_prompt = ([self.pad_token] * num_pad)
            full_prompt.extend(prompt)
            attention_mask[btc_idx][:num_pad] = 0.
            # try:
            full_prompt = torch.tensor(full_prompt, dtype=torch.long).unsqueeze(0)
            # except Exception as e:
            #     print(full_prompt)
            
            if res_prompt is None:
                res_prompt = full_prompt
            else:
                res_prompt = torch.cat((res_prompt, full_prompt), dim=0)
            real_idx += 1

        if len(res_image_idx) == 0:
            res_image_idx = None
        return {'video': res_video, 'prompt': res_prompt, 'char_image': res_image, 'ad': res_ad, 'mask': attention_mask, 'image_idx': res_image_idx, 'video_idx': res_video_idx} #
    
def get_json_dataset(args, is_train, model, eval_data=None, movie_id=None):
    if is_train:
        input_filename = args.train_data
        assert input_filename
        dataset = JsonDataset(
            input_filename,
            args.movie_feature_path,
            args.char_feature_path,
            model,
            args.if_special_prompt,
            if_train_drop=args.if_train_drop,
            char_prompt_type=args.char_prompt_type,
            previous_video_num=args.previous_video_num,
            if_video_split=args.if_video_split,
        )
    else:
        dataset = JsonDataset_EVAL(
            eval_data,
            args.movie_feature_path,
            args.char_feature_path,
            model,
            movie_id=movie_id,
            if_special_prompt=args.if_special_prompt,
            if_train_drop=args.if_train_drop,
            char_prompt_type=args.char_prompt_type,
            previous_video_num=args.previous_video_num,
            if_video_split = args.if_video_split,
        )

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    if is_train:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
            collate_fn=My_collate_fn_json(model.pad_token_id),
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size_val,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
            collate_fn=My_collate_fn_json_eval(model.pad_token_id),
        )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "json":
        return get_json_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    if args.train_data and args.dataset_type == "json":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, is_train=True, model=tokenizer)
    elif args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, model=tokenizer)
    if args.val_data and args.dataset_type == "json":
        with open(args.val_data) as f:
            eval_data = json.load(f)
        data["val"] = {}
        for k in eval_data.keys():
            temp = get_dataset_fn(args.val_data, args.dataset_type)(
                args, is_train=False, model=tokenizer, eval_data=eval_data[k], movie_id=k)
            data['val'][k] = temp
    elif args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    return data
