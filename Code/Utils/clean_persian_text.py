# -*- coding = utf-8 -*-
import re


class PersianTextCleaner:
    rep0 = {
        u'\u202c': ' ',
        u'\u202b': ' ',
        # u'\u200c': ' ',
        u'\xa0': ' ',
    }
    rep1 = {
        u'ـ': u'',
        u'÷': u'',
        u'`': u'',
        u'ٰ': u'',
        u'.': ' ',
        u'-': ' ',
        u'_': ' ',
        u'=': ' ',
        u'+': ' ',
        u'^': ' ',
        u'*': '',
        u'&': '',
        u'@': '',
        u'#': '',
        u'$': '',
        u'%': '',
        u'[': '',
        u']': '',
        u'{': '',
        u'}': '',
        u',': ' ',
        u'،': ' ',
        u'؟': ' ',
        u'!': ' ',
        u'?': ' ',
        u')': '',
        u'(': '',
        u':': ' ',
        u'"': '',
        u'\\': '',
        u'/': '',
        u'؛': '',
        u';': '',
        u'َ': '',
        u'ُ': '',
        u'ِ': '',
        u'ّ': '',
        u'ٌ': '',
        u'ٍ': '',
        u'ئ': u'ی',
        u'ي': u'ی',
        u'ة': u'ه',
        u'ء': u'',
        u'ك': u'ک',
        u'ْ': u'',
        u'|': u'',
        u'«': u'',
        u'»': u'',
        u'أ': u'ا',
        u'إ': u'ا',
        u'ؤ': u'و',
        u'×': u'',
        u'٪': u'',
        u'٬': u'',
        u'آ': u'ا',
        u'●': u'',
    }
    rep2 = {
        u'ـ': u'',
        u'÷': u'',
        u'`': u'',
        u'ٰ': u'',
        u'-': ' ',
        u'_': ' ',
        u'=': ' ',
        u'+': ' ',
        u'^': ' ',
        u'*': '',
        u'&': '',
        u'@': '',
        u'#': '',
        u'$': '',
        u'%': '',
        u'[': '',
        u']': '',
        u'{': '',
        u'}': '',
        u',': ' ',
        u'،': ' ',
        u')': '',
        u'(': '',
        u':': ' ',
        u'"': '',
        u'\\': '',
        u'/': '',
        u'َ': '',
        u'ُ': '',
        u'ِ': '',
        u'ّ': '',
        u'ٌ': '',
        u'ٍ': '',
        u'ئ': u'ی',
        u'ي': u'ی',
        u'ة': u'ه',
        u'ء': u'',
        u'ك': u'ک',
        u'ْ': u'',
        u'|': u'',
        u'«': u'',
        u'»': u'',
        u'أ': u'ا',
        u'إ': u'ا',
        u'ؤ': u'و',
        u'ً': u'',
        u'آ': u'ا',
        u'×': u'',
        u'٪': u'',
        u'٬': u'',
        u'●': u'',
        u'\'': u'',
        u'<': u' ',
        u'>': u' ',
    }
    rep3 = {
        u' . ': '.',
        u' ؟ ': '.',
        u' ? ': '.',
        u' ! ': '.',
        u' ؛ ': '',
        u' ; ': '',
    }
    rep4 = {
        u' .': '.',
        u'. ': '.',
        u' ؟': '.',
        u'؟ ': '.',
        u' ?': '.',
        u'? ': '.',
        u' !': '.',
        u'! ': '.',
        u' ؛': '.',
        u'؛ ': '.',
        u' ;': '.',
        u'; ': '.',
    }

    reg0 = dict((re.escape(k), v) for k, v in list(rep0.items()))
    reg1 = dict((re.escape(k), v) for k, v in list(rep1.items()))
    reg2 = dict((re.escape(k), v) for k, v in list(rep2.items()))
    reg3 = dict((re.escape(k), v) for k, v in list(rep3.items()))
    reg4 = dict((re.escape(k), v) for k, v in list(rep4.items()))

    pattern0 = re.compile("|".join(reg0.keys()))
    pattern1 = re.compile("|".join(reg1.keys()))
    pattern2 = re.compile("|".join(reg2.keys()))
    pattern3 = re.compile("|".join(reg3.keys()))
    pattern4 = re.compile("|".join(reg4.keys()))

    def __init__(self, special_rep):
        self.special_rep = special_rep
        special_reg = dict((re.escape(k), v) for k, v in list(special_rep.items()))
        self.special_reg = special_reg
        special_pattern = re.compile("|".join(special_reg.keys()))
        self.special_pattern = special_pattern

    def replace_numbers_with_smth(self, text):
        content = re.sub(u"[0-9۰-۹]*[.]?[0-9۰-۹]+", u"1", text)
        return content

    def replace_versions_with_smth(self, text):
        content = re.sub(u"[a-zA-z]*[0-9۰-۹]*[.]?[0-9۰-۹]+[a-zA-z]+", "eaae", text)
        content = re.sub(u"[a-zA-z]+[0-9۰-۹]*[.]?[0-9۰-۹]+[a-zA-z]*", "eaae", content)
        return content

    def pre_clean(self, text):
        content = text
        content = self.pattern0.sub(lambda m: self.reg0[re.escape(m.group(0))], content)
        content = re.sub('[a-zA-Z0-9]', ' ', content)
        return content

    def clean_text(self, text):
        content = self.pre_clean(text)
        content = " ".join(content.split())
        content = self.replace_versions_with_smth(content)
        content = self.replace_numbers_with_smth(content)
        content = self.pattern1.sub(lambda m: self.reg1[re.escape(m.group(0))], content)
        if len(self.special_rep) > 0:
            content = self.special_pattern.sub(lambda m: self.special_reg[re.escape(m.group(0))], content)

        content = re.sub(u"(eaae)+", u"eaae", content)
        content = re.sub(u"1+", u"1", content)
        return content

    def get_sentences(self, text):
        text = text.lower()

        content = self.pre_clean(text)
        content = ' '.join(content.splitlines())
        content = " ".join(content.split())
        content = self.replace_versions_with_smth(content)
        content = self.replace_numbers_with_smth(content)
        content = self.pattern3.sub(lambda m: self.reg3[re.escape(m.group(0))], content)
        content = self.pattern4.sub(lambda m: self.reg4[re.escape(m.group(0))], content)
        content = self.pattern2.sub(lambda m: self.reg2[re.escape(m.group(0))], content)
        content = re.sub(u"(eaae)+", u"eaae", content)
        content = re.sub(u"1+", u"1", content)

        # sentences = content.split('.')
        # non_empty_sentences = []
        # for sen in sentences:
        #     if (len(sen) > 0):
        #         non_empty_sentences.append(sen)
        return content
