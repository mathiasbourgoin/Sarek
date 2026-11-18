/**
 * process-rel-links.js
 * Author: Cory Gross
 * 
 * This simple jQuery script is called once the DOM structure has been
 * completely loaded. In order to make relative linking work correctly on
 * the Jekyll based GitHub Pages: we take each anchor on which the inner
 * anonymous function is executed, with the function context 'this' set to
 * the anchor element on which the function is being called. The function
 * simply determines if the extension is .md and removes it if so.
 **/

$(function () {
    $('a').each(function () {
        var ext = '.md';
        var href = $(this).attr('href');
        var position = href.length - ext.length;
        if (href.substring(position) === ext)
            $(this).attr('href', href.substring(0, position));
    });
});