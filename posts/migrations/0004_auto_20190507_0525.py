# Generated by Django 2.2.1 on 2019-05-07 05:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0003_auto_20190503_0811'),
    ]

    operations = [
        migrations.RenameField(
            model_name='hero',
            old_name='cover',
            new_name='image',
        ),
        migrations.RemoveField(
            model_name='hero',
            name='title',
        ),
    ]
