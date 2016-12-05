from django import forms

class SearchField(forms.Form):
    search = forms.CharField(label='',max_length=100)
