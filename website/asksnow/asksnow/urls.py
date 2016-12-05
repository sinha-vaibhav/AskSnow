from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),url(r'^$',include('basic.urls')),
    url(r'^basic/',include('basic.urls'))
]
