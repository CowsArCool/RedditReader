#
# Default Packages
from pydub.playback import play
from dataclasses import dataclass
from operator import sub
import re
import warnings
import logging
import json
import numpy as np
import time
import os

# Image Processing
import cv2
import cvzone
import textwrap
from PIL import ImageFont, ImageDraw, Image

# Audio Processing
from pydub import AudioSegment
import librosa.display

# Compositing
from moviepy.editor import VideoFileClip, AudioFileClip


# Internal Packages
# import sync_playback
from YoutubeAPI import YoutubeAPI
from reddit import *
import oddcast

# Requests
import requests
import hashlib
import shutil
import tempfile


DEFAULT_TAGS = ['Reddit', 'AskReddit']


class PostComponent:
    def __init__(self, text):
        self.IGNORED_CHARACTERS = {'>'}
        self.TTS_AGENTS = {
            'daniel': oddcast.generate_daniel
        }

        self.text, self.sentences = self._process_text(text)
        self.audioPath, self.nonSilentTime, self.srFreq = (
            self.build_audio(tts_agent='daniel'))

        self.failed = False
        if len(self.sentences) != len(self.nonSilentTime):
            logging.warning(
                'sentences length unmatched with non silent time'
                'discarding: {}\n'.format(repr(self)))
            self.failed = True
            return

        self.soundTrack = AudioSegment.from_file(
            self.audioPath)[: (self.nonSilentTime[-1][1]/self.srFreq)*1000]

    def __bool__(self):
        return not self.failed

    def __str__(self):
        return f'{self.name}: \"{self.text}\"'

    def __repr__(self):
        return f'{self.name} ({self.text=}, {self.failed=}, {self.audioPath=})'

    def _process_text(self, text):
        text = self._clean_text(text)
        sentences = self._split_sentences(text)
        return text, sentences

    def _clean_text(self, text):
        for char in self.IGNORED_CHARACTERS:
            text = text.replace(char, '')

        return text

    def _split_sentences(self, text):
        sentences = re.split('[.,;:!] |[\n\t()]', text)
        for blank_char in ['  ', ' ', '']:
            sentences = list(filter((blank_char).__ne__, sentences))

        return sentences

    def build_audio(self, tts_agent_generator=None, tts_agent='daniel'):
        # tts_agent_generator must return a path to the generated audio file
        assert tts_agent or tts_agent_generator, 'tts agent is unspecified'

        agent_function = (
            tts_agent_generator if tts_agent_generator else
            self.TTS_AGENTS[tts_agent]
        )

        audioPath = agent_function(self.text)

        with warnings.catch_warnings():
            # supress pysound warning
            warnings.simplefilter("ignore")
            x, sr = librosa.load(audioPath)
            nonSilentTime = librosa.effects.split(x, top_db=60)

        return audioPath, nonSilentTime, sr

    def align_audio_video(self):
        extraAudioLen = len(self.soundTrack)/1000 - self.frameGroup.duration
        if extraAudioLen > 0:
            self.frameGroup.resize(lengthDelta=extraAudioLen, location='end')

        elif extraAudioLen < 0:
            self.soundTrack = self.soundTrack + AudioSegment.silent(
                duration=abs(extraAudioLen)*1000)


class Comment (PostComponent):
    def __init__(self, reddit_comment, config):
        self.name = 'Comment'
        text = reddit_comment.body
        super().__init__(text)
        if self.failed:
            return

        self.frameGroup = CommentFrameGroup(
            comment=reddit_comment,
            nonSilentTime=self.nonSilentTime,
            sentences=self.sentences,
            sr=self.srFreq,
            config=config
        )

        self.frameGroup.insert_trailing_space(config.spacingTime)
        self.soundTrack = self.soundTrack + \
            AudioSegment.silent(config.spacingTime*1000)

        self.align_audio_video()

        self.frameGroup.compile_frames()

    def name(self):
        return 'Comment'


class Title(PostComponent):
    def __init__(self, reddit_post, config):
        self.name = 'Title'
        text = reddit_post.title

        self.display_info = {
            'subreddit': reddit_post.subreddit.display_name,
            'title': text
        }

        super().__init__(text)
        if self.failed:
            return

        self.frameGroup = TitleFrameGroup(
            reddit_post,
            nonSilentTime=self.nonSilentTime,
            sentences=self.sentences,
            sr=self.srFreq,
            config=config
        )

        self.align_audio_video()

        self.frameGroup.compile_frames()


class Transition:
    def __init__(self, config):
        self.name = 'Transition'
        transitionDir = os.path.join(
            RUNNING_DIR, 'transitions', config.transitionDir)
        transitionAudioPath = os.path.join(transitionDir, 'audio.mp3')
        transitionFramesDir = os.path.join(transitionDir, 'frames')

        self.soundTrack = AudioSegment.from_file(transitionAudioPath)
        self.soundTrack += config.transitionVolumeDelta
        self.frameGroup = TransitionFrameGroup(config, transitionFramesDir)

        self.clip_audio()

        self.frameGroup.compile_frames()

    def clip_audio(self):
        extraAudioLen = len(self.soundTrack)/1000 - self.frameGroup.duration
        if extraAudioLen > 0:
            self.soundTrack = self.soundTrack[
                : len(self.soundTrack)-int(extraAudioLen*1000)]

        elif extraAudioLen < 0:
            self.frameGroup.resize(lengthDelta=extraAudioLen)

        extraAudioLen = len(self.soundTrack)/1000 - self.frameGroup.duration

    def __repr__(self):
        return f'Transition({self.frameGroup.duration=}, {len(self.soundTrack)/1000=})'


#
def build_tempfile(hashable, prefix=None, suffix=None):
    hashed = hashlib.md5(hashable.encode()).hexdigest()
    return os.path.join(
        tempfile.gettempdir(),
        '{}-{}{}'.format(
            prefix,
            hashed,
            suffix
        )
    )


#
class Frame:
    def __init__(self, frame, duration: float | None = None):
        # frame is a numpy array

        self.frame = frame
        self._duration = duration

    def __str__(self):
        return f'Frame (duration = {self.duration} seconds, frame = {self.frame})'

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self._duration = float(value)

    def numpy(self):
        return self.frame

    def compile(self, fps):
        # depreciated
        return [
            [self.frame]
            * self.duration * fps
        ]

    def display(self):
        cv2.imshow(f'Frame: {self.duration} seconds', self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def overlay_masked_image(self,  url=None, image=None, resize=None, radius=25, position=None,):
        # url is a link to the image
        # image is obviously just that

        assert url or image, 'unspecified input image/url'
        assert position is not None, 'position argument must be specified'

        overlayImg = image
        if url:
            overlayImg = self.download_image(url)

        if resize:
            overlayImg = self.resize_img(overlayImg, resize)

        if overlayImg.shape[2] == 3:
            hh, ww = overlayImg.shape[:2]
            xc = hh // 2
            yc = ww // 2
            radius = radius

            mask = np.zeros_like(overlayImg)
            mask = cv2.circle(mask, (xc, yc), radius, (255, 255, 255), -1)

            overlayImg = np.concatenate((
                overlayImg,
                np.expand_dims(mask[:, :, 0], axis=2)
            ), axis=2)

        self.frame = cvzone.overlayPNG(
            self.frame,
            overlayImg,
            position
        )

    def display_text(self, text, position, font, fontFill):
        img_pil = Image.fromarray(self.frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text,
                  font=font, fill=fontFill)

        self.frame = np.array(img_pil)

    def display_wrapping_text(
        self, text: str, positions: list,
        font, fontFill: str | int,
        textWrapCharacterLimit=80,
        maxLinesInFrame=20
    ):
        wrapped_text = textwrap.wrap(
            text, width=textWrapCharacterLimit,
            fix_sentence_endings=True
        )

        if len(list(wrapped_text)) > maxLinesInFrame:
            wrapped_text = wrapped_text[maxLinesInFrame-1:]

        for i, line in enumerate(wrapped_text):
            textsize = font.getsize(line)
            lineSpace = textsize[1] + 10

            x = positions[0]
            y = positions[1] + i * lineSpace

            self.display_text(line, (x, y), font, fontFill)

    @staticmethod
    def download_image(url, **kwargs):
        # kwargs contains prefix and suffix
        temp_path = build_tempfile(url, **kwargs)

        res = requests.get(url, stream=True)
        if res.status_code == 200:
            with open(temp_path, 'wb') as f:
                shutil.copyfileobj(res.raw, f)

        return cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def resize_img(img, scale_percent):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class CompiledFrames:
    frames: list

    def __getitem__(self, idx):
        return self.compiled_frames[idx]

    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        return f'CompiledFrames(n_frames = {len(self)}, size = {self.frames[0].shape})'


class FrameGroup:
    def __init__(self, config):
        self.template = Frame(
            cv2.imread(
                os.path.join(
                    RUNNING_DIR,
                    'templates',
                    config.filename
                )
            )
        )

        self.frames = list()
        self._compiled_frames = None

        self.config = config

    def truncate_or_pad(self, lengthDelta, location='end'):
        if location == 'end':
            assert self.frames[-1].duration > abs(lengthDelta) or lengthDelta >= 0, (
                'lengthDelta must be less than final frame length when truncating')

            self.frames[-1].duration += lengthDelta

        elif location == 'start':
            assert self.frames[0].duration > abs(lengthDelta) or lengthDelta >= 0, (
                'lengthDelta must be less than first frame length when truncating')

            self.frames[0] += lengthDelta

    def resize(self, lengthDelta=None, length=None, location='end'):
        assert lengthDelta or length
        # all measurements are seconds
        if lengthDelta:
            self.truncate_or_pad(lengthDelta, location)

        elif length:
            self.truncate_or_pad(self.duration - length, location)

    def compile_frames(self):
        compiled_frames = []
        for frame in self.frames:
            compiled_frames.extend(
                [frame.numpy()]
                * int(self.config.fps * frame.duration)
                # frame.compile(fps)
            )

        self._compiled_frames = CompiledFrames(compiled_frames)

    @property
    def compiled_frames(self):
        if not self._compiled_frames:
            raise ValueError(
                'compile_frames method must be called '
                'before accessing compiled_frames')

        return self._compiled_frames

    @property
    def duration(self):
        return sum([frame.duration for frame in self.frames])

    def __len__(self):
        # returns seconds seconds
        return len(self.frames)

    def __repr__(self):
        return f'Frame Group (duration={self.duration}, n_frames={len(self)})'


class TitleFrameGroup (FrameGroup):
    def __init__(
            self, post, nonSilentTime,
            sentences, sr, config):

        super().__init__(config)

        self.subreddit = post.subreddit
        self.display_subredit()

        frame = Frame(self.template.numpy())
        frame.display_wrapping_text(
            post.title,
            self.config.textPosition,
            self.config.textFont,
            (255, 255, 255, 0),
            self.config.textWrapCharacterLimit,
            self.config.maxLinesInFrame
        )

        frame.duration = nonSilentTime[-1][-1]/sr

        self.frames.append(frame)

    def display_subredit(self):
        self._display_sub_text()
        self._display_sub_icon()

    def _display_sub_text(self):
        self.template.display_text(
            '{}'.format(self.subreddit.display_name),
            position=self.config.authorTextPosition,
            font=self.config.authorFont,
            fontFill=self.config.authorFontFill
        )

    def _display_sub_icon(self):
        self.template.overlay_masked_image(
            url=self.subreddit.icon_img,
            resize=self.config.authorImgScale,
            radius=self.config.authorImgRadius,
            position=self.config.authorImgPosition
        )


class CommentFrameGroup (FrameGroup):
    def __init__(
        self,
        comment,
        nonSilentTime,
        sentences, sr,
        config
    ):

        super().__init__(config)

        self.author = comment.author
        self.display_user_profile()
        sentenceDisplayTimes = np.append(
            nonSilentTime[:, 0], (nonSilentTime[-1, 1]))

        # space before tts agent starts speaking
        self.template.duration = sentenceDisplayTimes[0]/sr
        self.frames.append(self.template)

        for i in range(len(sentences)):
            text = '. '.join(sentences[:i+1])

            frame = Frame(self.template.numpy())

            frame.duration = (
                sentenceDisplayTimes[i+1]-sentenceDisplayTimes[i])/sr
            frame.display_wrapping_text(
                text,
                self.config.textPosition,
                self.config.textFont,
                (255, 255, 255, 0),
                self.config.textWrapCharacterLimit,
                self.config.maxLinesInFrame
            )

            self.frames.append(frame)

    def _display_author_text(self):
        authorName = self.author if self.author else '[deleted]'

        self.template.display_text(
            'u/{}'.format(authorName),
            position=self.config.authorTextPosition,
            font=self.config.authorFont,
            fontFill=self.config.authorFontFill
        )

    def _display_author_img(self):
        default_icon_img = (
            'https://www.redditstatic.com/'
            'avatars/defaults/v2/avatar_default_5.png')
        icon_img = getattr(
            self.author,
            'icon_img',
            default_icon_img
        )

        self.template.overlay_masked_image(
            url=icon_img,
            resize=self.config.authorImgScale,
            radius=self.config.authorImgRadius,
            position=self.config.authorImgPosition
        )

    def display_user_profile(self):
        self._display_author_text()
        self._display_author_img()

    def insert_trailing_space(self, time):
        self.frames[-1].duration += time


class TransitionFrameGroup (FrameGroup):
    def __init__(self, config, framesDir):
        self.config = config

        frame_ratio = int(round(
            self.config.transitionFramesNativeFPS/self.config.fps))
        self._compiled_frames = None
        self.frames = [
            cv2.imread(os.path.join(framesDir, filename))
            for filename in os.listdir(framesDir)
        ][:: frame_ratio]

    @property
    def duration(self):
        return len(self.frames) / self.config.fps

    def compile_frames(self):
        self._compiled_frames = CompiledFrames(self.frames)

    def truncate_or_pad(self, lengthDelta, location='end'):
        framesDelta = int(round(self.config.fps * lengthDelta))
        if location == 'end':
            if lengthDelta < 0:
                self.frames = self.frames[: framesDelta]

            elif lengthDelta > 0:
                self.frames.extend([self.frames[-1]] * framesDelta)

        elif location == 'start':
            if lengthDelta < 0:
                self.frames = self.frames[abs(framesDelta):]

            elif lengthDelta > 0:
                self.frames[:0] = ([self.frames[0]] * framesDelta)

    def __len__(self):
        return len(self.frames)


class ConfigHandler:
    def __init__(self, configPath, platform):
        GLOBAL_ATTRS = [
            'spacingTime',
            'fps'
        ]

        with open(configPath) as f:
            self.template_attrs = json.load(f)[platform]

        self.globalConfigs = {key: self.template_attrs[key]
                              for key in GLOBAL_ATTRS}

    def GenerateConfig(self, frameType):
        return Config(**self.template_attrs[frameType], **self.globalConfigs)

    def __getattr__(self, key):
        return self.globalConfigs[key]

    def __repr__(self):
        return 'ConfigHandler (dict: {})'.format(
            ', '.join(
                [
                    f'{key} = {value}' for key, value in {
                        **self.template_attrs, **self.globalConfigs}.items()
                ]
            )
        )


@dataclass(repr=True, frozen=True)
class Config:
    # globals
    spacingTime: float
    fps: int

    # positionals

    filename: str = None

    fontStyle = 'verdana'

    textFontSize: int = None
    textFontFill: int = None
    authorFontSize: int = None
    authorFontFill: int = None

    authorImgPosition: list = None
    authorTextPosition: list = None
    authorImgScale: int = None
    authorImgRadius: int = None

    textPosition: list = None
    textWrapCharacterLimit: int = None
    maxLinesInFrame: int = None

    # kwwargs
    score: dict = None

    transitionVolumeDelta: int = None
    transitionDir: str = None
    transitionFramesNativeFPS: int = None

    fonts = {key: None for key in
             ['textFont', 'authorFont', 'scoreFont']}

    def _getFont(self, name=str, size=int, font: str = 'verdana'):
        if self.fonts[name] is not None:
            return self.fonts[name]
        font = ImageFont.truetype(
            font=font,
            size=size
        )

        self.fonts[name] = font
        return font

    @property
    def textFont(self):
        return self._getFont(name='textFont', size=self.textFontSize)

    @property
    def authorFont(self):
        return self._getFont(name='authorFont', size=self.authorFontSize)

    @property
    def scoreFont(self):
        return self._getFont(name='scoreFont', **self.score['font'])


#
class Video:
    def __init__(self, components, fps):
        """Compiles video from frames and AudioSegments

        Args:
            components (list[PostComponent]): list of all post components
                                              sorted by display order
        """
        audioSegments = list()
        videoFrames = list()
        for component in components:
            videoFrames.extend(component.frameGroup.compiled_frames.frames)
            audioSegments.append(component.soundTrack)

        soundTrack = sum(audioSegments)

        self.components = components
        self.fps = fps

        self._write_frames(videoFrames)
        self._merge_audio(soundTrack)

    def _write_frames(self, frames):
        unixTime = str(int(time.time()))
        temp_path = build_tempfile(unixTime, prefix='video', suffix='.avi')

        frame = frames[0]
        height, width, layers = frame.shape
        video = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), self.fps, (width, height))

        for image in frames:
            video.write(image)

        cv2.destroyAllWindows()
        video.release()

        self.videoPath = temp_path

    def _merge_audio(self, sound):
        temp_path = build_tempfile(
            str(int(time.time())))
        sound.export(temp_path, format='mp3')

        self.audioPath = temp_path

    def composite_audio_video(self):
        video_clip = VideoFileClip(self.videoPath)
        audio_clip = AudioFileClip(self.audioPath)
        self.composite = video_clip.set_audio(audio_clip)

    def export(self, video_name=None):
        assert hasattr(self, 'composite'), (
            'Must composite audio video before exporting')

        display_info = self.components[0].display_info
        if not video_name:
            video_name = (
                f'{display_info["subreddit"]}-'
                f'{display_info["title"][:20]}- '
                f'{str(int(time.time()))[-4:]}'
            )

        composite_path = os.path.join(
            RUNNING_DIR, 'videos', f'{video_name}.mp4'
        )

        self.composite.write_videofile(composite_path, fps=self.fps)

        self._path = composite_path

    @property
    def path(self):
        if (outpath := getattr(self, '_path', None)) is not None:
            return outpath

        else:
            raise AttributeError(
                'Video object has not been exported and ' +
                'does not have an output path '
            )


class InvalidPostTitle (Exception):
    pass


def post_av_duration_dif(components):
    soundTrackDuration = sum([len(component.soundTrack)/1000
                              for component in components])
    videoDuration = sum([len(component.frameGroup.compiled_frames) / component.frameGroup.config.fps
                         for component in components])

    return soundTrackDuration - videoDuration


def align_post_av(components):
    soundTrackDuration = sum([len(component.soundTrack)/1000
                              for component in components])
    videoDuration = sum([component.frameGroup.duration
                         for component in components])

    dd = soundTrackDuration - videoDuration

    durationDifference = post_av_duration_dif(components)
    # assert dd == durationDifference, f'{dd=} {durationDifference=}'

    if durationDifference > 0:
        logging.debug(f'removing {durationDifference}')
        components[-1].soundTrack = components[-1].soundTrack[:-
                                                              durationDifference*1000]

    elif durationDifference < 0:
        logging.debug(f'adding {durationDifference}')
        components[-1].soundTrack += AudioSegment.silent(
            durationDifference*1000)


def upload_video(videoPath, title, description='', visibility='private'):
    api = YoutubeAPI(
        user_data_dir=r"C:\Users\micha\AppData\Local\Google\Chrome\User Data")

    api.uploadVideo(videoPath, title=title,
                    description=description, visibility=visibility)


def build_video(subreddit, platform, n_posts, n_comments):
    global RUNNING_DIR

    RUNNING_DIR = os.path.dirname(os.path.realpath(__file__))

    reddit = RedditViewer()
    reddit.load_viewed_posts(RUNNING_DIR)

    globalConfig = ConfigHandler(
        os.path.join(RUNNING_DIR, 'template_positions.json'),
        platform
    )

    total_components = list()
    for i in range(n_posts):
        try:
            av_components = build_av_components(
                reddit=reddit,
                globalConfig=globalConfig,
                subreddit=subreddit,
                n_comments=n_comments,
                trailing_transition=False
            )
            align_post_av(av_components)
            logging.debug(
                f'post diff: {post_av_duration_dif(components=av_components)} at {i}')
            logging.debug(
                f'overall diff: {post_av_duration_dif(components=total_components)} at {i}')

        except Exception as e:
            raise e
            logging.error(f'{e.__class__.__name__} on post {i}\n')
            continue

        total_components.extend(av_components)

    logging.info('Completed Building AV Components')

    video = Video(total_components, globalConfig.fps)
    video.composite_audio_video()
    video.export(
        f'CompilationVideo-{subreddit}-{str(int(time.time()))[-4:]}'
    )

    return video


def build_av_components(
    reddit,
    globalConfig,
    subreddit,
    n_comments=5,
    trailing_transition=False
):
    submission = reddit.query_sub(
        subreddit,
        disallowed_flags=['stickied'],
        sort='top',
        sort_kwargs={'time_filter': 'day'}
    )[0]
    reddit.save_viewed_posts(RUNNING_DIR)

    submission.comment_sort = 'top'
    selected_comments, _ = filter_comments(
        submission,
        min_len=0,
        max_len=300,
        len_variability=0,
        sort='best',
        n=n_comments
    )
    selected_comments = selected_comments.tolist()

    titleConfig = globalConfig.GenerateConfig('title')
    commentConfig = globalConfig.GenerateConfig('comment')
    transitionConfig = globalConfig.GenerateConfig('transition')

    title = Title(submission, titleConfig)
    if not title:
        raise InvalidPostTitle

    content_components = list()
    for reddit_comment in selected_comments:
        comment = Comment(reddit_comment, commentConfig)
        transition = Transition(transitionConfig)
        if comment:
            content_components.append(transition)
            content_components.append(comment)

    components = [*[title], *content_components]

    if trailing_transition:
        components.append(Transition(transitionConfig))

    logging.info(f'Built components for submission {submission.title}')

    return components


def main(platform, subreddit, N_POSTS):
    global RUNNING_DIR
    # RUNNING_DIR = os.path.dirname(os.path.realpath(__file__))
    RUNNING_DIR = r'C:\Code\Random_Shit\RedditReaderV2'

    reddit = RedditViewer()
    reddit.load_viewed_posts(RUNNING_DIR)

    submission = reddit.query_sub(
        subreddit,
        disallowed_flags=['stickied'],
        sort='top',
        sort_kwargs={'time_filter': 'day'}
    )[0]
    reddit.save_viewed_posts(RUNNING_DIR)

    submission.comment_sort = 'top'
    selected_comments, _ = filter_comments(
        submission,
        min_len=0,
        max_len=300,
        len_variability=0,
        sort='best',
        n=N_POSTS
    )
    selected_comments = selected_comments.tolist()

    globalConfig = ConfigHandler(
        os.path.join(RUNNING_DIR, 'template_positions.json'),
        platform
    )

    titleConfig = globalConfig.GenerateConfig('title')
    commentConfig = globalConfig.GenerateConfig('comment')
    transitionConfig = globalConfig.GenerateConfig('transition')

    title = Title(submission, titleConfig)
    if not title:
        return

    content_components = list()
    for reddit_comment in selected_comments:
        comment = Comment(reddit_comment, commentConfig)
        transition = Transition(transitionConfig)
        if comment:
            content_components.append(transition)
            content_components.append(comment)

    components = [*[title], *content_components]

    video = Video(components, fps=globalConfig.fps)
    video.composite_audio_video()
    video.export()


if __name__ == '__main__':
    outfile = 'logging_output.txt'
    open(outfile, 'w+').close()
    logging.basicConfig(
        level=logging.INFO,
        # filename=outfile
    )

    subreddit = 'AskReddit'
    video = build_video(
        subreddit=subreddit,
        platform='standard',
        n_posts=1,
        n_comments=2
    )

    logging.info(f'Video Built at {video.path}')

    firstTitle = [
        component for component in video.components
        if isinstance(component, Title)][0].text
    videoTitle = '{} | {}'.format(firstTitle, subreddit)
    upload_video(video.path, title=videoTitle,
                 description=', '.join(DEFAULT_TAGS))
