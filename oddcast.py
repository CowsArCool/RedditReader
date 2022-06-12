from urllib.parse import urlencode
import hashlib
import os
import requests
import sys
import tempfile


def build_hash(text, engine, voice, language, fx=None, fx_level=None):
    fragments = [
        "<engineID>%s</engineID>" % engine,
        "<voiceID>%s</voiceID>" % voice,
        "<langID>%s</langID>" % language,
        ("<FX>%s%s</FX>" % (fx, fx_level)) if fx else '',
        "<ext>mp3</ext>",
        text,
    ]

    return hashlib.md5(''.join(fragments).encode('utf-8')).hexdigest()


def get_tts_url(text, engine, voice, language, fx=None, fx_level=None):
    hash = build_hash(**locals())
    params = [
        ('engine', engine),
        ('language', language),
        ('voice', voice),
        ('text', text),
        ('useUTF8', 1),
        ('fx_type', fx),
        ('fx_level', fx_level),
    ]
    params = [(key, value) for (key, value) in params if (key and value)]

    return 'http://cache-a.oddcast.com/c_fs/%s.mp3?%s' % (
        hash,
        urlencode(params),
    )


def download(text, engine, language, voice, fx=None, fx_level=None):
    url = get_tts_url(**locals())
    # print(url)

    name = 'longtest'
    temp_name = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'tests',
        f'{name}_{voice}.mp3'
    )

    temp_name = os.path.join(
        tempfile.gettempdir(),
        'tts-{}.mp3'.format(
            hashlib.md5(url.encode()).hexdigest()
        )
    )

    # temp_name = os.path.join(tempfile.gettempdir(), 'tts-testbruh.mp3') #  % hashlib.md5(url).hexdigest()
    # temp_name = r'C:\Code\Random_Shit\RedditReader\tests\test1.mp3'
    if os.path.exists(temp_name):
        os.remove(temp_name)

    with open(temp_name, 'wb') as outf:
        resp = requests.get(url)
        resp.raise_for_status()
        outf.write(resp.content)

    return temp_name


def test_hash():
    assert build_hash('nnep', engine=4, language=23,
                      voice=1) == '8663a3e4e10637477864d8252704a38d'


def generate_daniel(text):
    tmp = download(
        text=text,
        engine=4,
        language=1,
        voice=5,
    )
    # print(f'generating {text} with daniel')
    # print(tmp)

    return tmp


if __name__ == '__main__':
    text = r'hippity hoppity boopity boppity'
    output = generate_daniel(text)
    print(output)

    from pydub import AudioSegment
    from pydub.playback import play

    play(AudioSegment.from_file(output))

    # tmp = download(
    #     text=(' '.join(sys.argv[1:])
    #           or text),
    #     engine=4,
    #     language=1,
    #     voice=5,
    # )

    # pygame.mixer.init()
    # pygame.mixer.music.load(tmp)
    # pygame.mixer.music.play()
    # player = ('afplay' if sys.platform == 'darwin' else 'ffplay')
    # os.system('%s %s' % (player, tmp))
