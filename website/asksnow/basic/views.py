from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from .forms import SearchField
import json
import requests
import re
import logging
logger = logging.getLogger(__name__)
parameter_url = 'http://52.11.71.138:8983/solr/IRP4/select?q=_QUERY_&rows=20&wt=json'

def sanatize(instr):
	#escape characters which solr cant handle in query
	for s in '+-!(){}[]^\"~*?:':
		instr=instr.replace(s,'\\'+s)
	return instr
def index(request):
	return render(request,'basic/basictemplate.html',{'form':SearchField()})

def results(request):
	if request.method=='POST':
		form =SearchField(request.POST)
		if form.is_valid():
			searchStr=form.cleaned_data['search']
	else:
		searchStr=request.GET.get('q')
	query=sanatize(searchStr)
	data = requests.get(parameter_url.replace('_QUERY_',query))
	if data.status_code !=200:
		logger.error(data)
		return HttpResponse("Error in solr.\n")
	docs = data.json()['response']['docs']
	results=[]

	for doc in docs:
		results.append(doc['tweet_text'][0])
	return render(request,'basic/secondPage.html',{'results':results[0:5]})
	#return HttpResponse("Hello, world. You're at the polls index.\n")
