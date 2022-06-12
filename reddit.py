import praw
import pandas as pd
import numpy as np
import json
import os


class RedditViewer:
    def __init__(self,):
        self.reddit = praw.Reddit(
            client_id="1ZVjeoaDgmug0w",
            client_secret="DW_GfyDt2RP8qWCTjmHemvA8iKl3uA",
            user_agent="by u/SlimyBananas",
        )

        self.viewed_posts = {}

    def query_sub(self, sub, sort='hot', n=1, disallowed_flags=[], sort_kwargs={}):
        if sub not in list(self.viewed_posts.keys()):
            self.viewed_posts[sub] = list()

        submission_generator = getattr(
            self.reddit.subreddit(sub), sort)(**sort_kwargs)

        unique_submissions = []
        while len(unique_submissions) < n:
            submission = next(submission_generator)
            if (
                    submission.id not in self.viewed_posts[sub] and
                    not any([
                        getattr(submission, flag)
                        for flag in disallowed_flags
                    ])):

                unique_submissions.append(submission)
                self.viewed_posts[sub].append(submission.id)

        return unique_submissions

    def fetch_sub_header(self, sub, attrs=['icon_img']):
        r = self.reddit.subreddit(sub)
        return [getattr(r, attr) for attr in attrs]

    def reset_viewed_posts(self):
        self.viewed_posts = {
            key: list() for key in self.viewed_posts.keys()
        }

    def save_viewed_posts(self, save_dir):
        save_path = os.path.join(save_dir, 'reddit_viewed_posts.json')
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path, 'w') as outfile:
            json.dump(self.viewed_posts, outfile)

    def load_viewed_posts(self, load_dir):
        load_path = os.path.join(load_dir, 'reddit_viewed_posts.json')
        if os.path.exists(load_path):
            with open(load_path) as infile:
                self.viewed_posts = json.load(infile)
        else:
            print('No such file {}'.format(load_path))

    def __len__(self):
        return int(np.sum([len(value) for value in self.viewed_posts.values()]))

    def __str__(self):
        return '{} (Viewed Posts: {} from {} {})'.format(
            self.__class__.__name__, self.__len__(), len(self.viewed_posts),
            'sub' if len(self.viewed_posts) == 1 else 'subs')


def filter_comments(submission, min_len=0, max_len=np.inf, len_variability=0, n=1, sort='best'):
    submission.comment_sort = sort
    eligibile_comments = np.zeros((n, 2), dtype=object)
    submission.comment_sort

    i = 0
    for comment in iter(submission.comments):
        if not isinstance(comment, praw.models.Comment):
            return eligibile_comments[:, 0], 'failed'

        distance_from_min_variability = max(0,
                                            len_variability - np.std(
                                                eligibile_comments[:, 1]
                                            ))

        comment_distance_from_mean = np.absolute(
            len(comment.body)-np.mean(
                eligibile_comments[:, 1]
            )
        )

        if (max_len >= len(comment.body) >= min_len
                and comment_distance_from_mean >= distance_from_min_variability
                and comment.author is not None):
            eligibile_comments[i] = (comment, len(comment.body))
            i += 1

            if i >= n:
                return eligibile_comments[:, 0], 'success'


if __name__ == '__main__':
    reddit = RedditViewer()

    submission = reddit.query_sub(
        'AskReddit', disallowed_flags=['stickied'])[0]

    submission.comment_sort = 'top'
    selected_comments, _ = filter_comments(
        submission, min_len=0, max_len=10000, len_variability=0, n=3, sort='best')

    reddit.fetch_sub_header('AskReddit')
