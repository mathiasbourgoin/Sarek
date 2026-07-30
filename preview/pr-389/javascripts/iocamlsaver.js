var saveAs = saveAs
  // IE 10+ (native saveAs)
  || (typeof navigator !== "undefined" &&
      navigator.msSaveOrOpenBlob && navigator.msSaveOrOpenBlob.bind(navigator))
  // Everyone else
  || (function(view) {
      "use strict";
      // IE <10 is explicitly unsupported
      if (typeof navigator !== "undefined" &&
	      /MSIE [1-9]\./.test(navigator.userAgent)) {
	  return;
	  }
      var
        doc = view.document
        // only get URL when necessary in case Blob.js hasn't overridden it yet
      , get_URL = function() {
	  return view.URL || view.webkitURL || view;
	  }
      , save_link = doc.createElementNS("http://www.w3.org/1999/xhtml", "a")
      , can_use_save_link = !view.externalHost && "download" in save_link
      , click = function(node) {
	  var event = doc.createEvent("MouseEvents");
	  event.initMouseEvent(
	      "click", true, false, view, 0, 0, 0, 0, 0
	      , false, false, false, false, 0, null
	      );
	  node.dispatchEvent(event);
	  }
      , webkit_req_fs = view.webkitRequestFileSystem
      , req_fs = view.requestFileSystem || webkit_req_fs || view.mozRequestFileSystem
      , throw_outside = function(ex) {
	  (view.setImmediate || view.setTimeout)(function() {
	      throw ex;
	      }, 0);
	  }
      , force_saveable_type = "application/octet-stream"
      , fs_min_size = 0
      , deletion_queue = []
      , process_deletion_queue = function() {
	  var i = deletion_queue.length;
	  while (i--) {
	      var file = deletion_queue[i];
	      if (typeof file === "string") { // file is an object URL
		  get_URL().revokeObjectURL(file);
		  } else { // file is a File
		      file.remove();
		      }
	      }
	  deletion_queue.length = 0; // clear queue
	  }
      , dispatch = function(filesaver, event_types, event) {
	  event_types = [].concat(event_types);
	  var i = event_types.length;
	  while (i--) {
	      var listener = filesaver["on" + event_types[i]];
	      if (typeof listener === "function") {
		  try {
		      listener.call(filesaver, event || filesaver);
		      } catch (ex) {
			  throw_outside(ex);
			  }
		  }
	      }
	  }
      , FileSaver = function(blob, name) {
	  // First try a.download, then web filesystem, then object URLs
	  var
	    filesaver = this
	  , type = blob.type
	  , blob_changed = false
	  , object_url
	  , target_view
	  , get_object_url = function() {
	      var object_url = get_URL().createObjectURL(blob);
	      deletion_queue.push(object_url);
	      return object_url;
	      }
	  , dispatch_all = function() {
	      dispatch(filesaver, "writestart progress write writeend".split(" "));
	      }
	  // on any filesys errors revert to saving with object URLs
	  , fs_error = function() {
	      // don't create more object URLs than needed
	      if (blob_changed || !object_url) {
		  object_url = get_object_url(blob);
		  }
	      if (target_view) {
		  target_view.location.href = object_url;
		  } else {
		      window.open(object_url, "_blank");
		      }
	      filesaver.readyState = filesaver.DONE;
	      dispatch_all();
	      }
	  , abortable = function(func) {
	      return function() {
		  if (filesaver.readyState !== filesaver.DONE) {
		      return func.apply(this, arguments);
		      }
		  };
	      }
	  , create_if_not_found = {create: true, exclusive: false}
	  , slice
	  ;
	  filesaver.readyState = filesaver.INIT;
	  if (!name) {
	      name = "download";
	      }
	  if (can_use_save_link) {
	      object_url = get_object_url(blob);
	      save_link.href = object_url;
	      save_link.download = name;
	      click(save_link);
	      filesaver.readyState = filesaver.DONE;
	      dispatch_all();
	      return;
	      }
	  // Object and web filesystem URLs have a problem saving in Google Chrome when
	  // viewed in a tab, so I force save with application/octet-stream
	  // http://code.google.com/p/chromium/issues/detail?id=91158
	  if (view.chrome && type && type !== force_saveable_type) {
	      slice = blob.slice || blob.webkitSlice;
	      blob = slice.call(blob, 0, blob.size, force_saveable_type);
	      blob_changed = true;
	      }
	  // Since I can't be sure that the guessed media type will trigger a download
	  // in WebKit, I append .download to the filename.
	  // https://bugs.webkit.org/show_bug.cgi?id=65440
	  if (webkit_req_fs && name !== "download") {
	      name += ".download";
	      }
	  if (type === force_saveable_type || webkit_req_fs) {
	      target_view = view;
	      }
	  if (!req_fs) {
	      fs_error();
	      return;
	      }
	  fs_min_size += blob.size;
	  req_fs(view.TEMPORARY, fs_min_size, abortable(function(fs) {
	      fs.root.getDirectory("saved", create_if_not_found, abortable(function(dir) {
		  var save = function() {
		      dir.getFile(name, create_if_not_found, abortable(function(file) {
			  file.createWriter(abortable(function(writer) {
			      writer.onwriteend = function(event) {
				  target_view.location.href = file.toURL();
				  deletion_queue.push(file);
				  filesaver.readyState = filesaver.DONE;
				  dispatch(filesaver, "writeend", event);
				  };
			      writer.onerror = function() {
				  var error = writer.error;
				  if (error.code !== error.ABORT_ERR) {
				      fs_error();
				      }
				  };
			      "writestart progress write abort".split(" ").forEach(function(event) {
				  writer["on" + event] = filesaver["on" + event];
				  });
			      writer.write(blob);
			      filesaver.abort = function() {
				  writer.abort();
				  filesaver.readyState = filesaver.DONE;
				  };
			      filesaver.readyState = filesaver.WRITING;
			      }), fs_error);
			  }), fs_error);
		      };
		  dir.getFile(name, {create: false}, abortable(function(file) {
		      // delete file if it already exists
		      file.remove();
		      save();
		      }), abortable(function(ex) {
			  if (ex.code === ex.NOT_FOUND_ERR) {
			      save();
			      } else {
				  fs_error();
				  }
			  }));
		  }), fs_error);
	      }), fs_error);
	  }
      , FS_proto = FileSaver.prototype
      , saveAs = function(blob, name) {
	  return new FileSaver(blob, name);
	  }
      ;
      FS_proto.abort = function() {
	  var filesaver = this;
	  filesaver.readyState = filesaver.DONE;
	  dispatch(filesaver, "abort");
	  };
      FS_proto.readyState = FS_proto.INIT = 0;
      FS_proto.WRITING = 1;
      FS_proto.DONE = 2;

      FS_proto.error =
	  FS_proto.onwritestart =
	  FS_proto.onprogress =
	  FS_proto.onwrite =
	  FS_proto.onabort =
	  FS_proto.onerror =
	  FS_proto.onwriteend =
	  null;

      view.addEventListener("unload", process_deletion_queue, false);
      saveAs.unload = function() {
	  process_deletion_queue();
	  view.removeEventListener("unload", process_deletion_queue, false);
	  };
      return saveAs;
}(
       typeof self !== "undefined" && self
    || typeof window !== "undefined" && window
    || this.content
));
// `self` is undefined in Firefox for Android content script context
// while `this` is nsIContentFrameMessageManager
// with an attribute `content` that corresponds to the window

/*if (typeof module !== "undefined" && module !== null) {
  module.exports = saveAs;
} else if ((typeof define !== "undefined" && define !== null) && (define.amd != null)) {
  define([], function() {
    return saveAs;
  });
}*/


function saveAsOcaml () {
/*    var nodes = document.getElementsByClassName("CodeMirror cm-s-ipython");*/

/*    var ocamlcode = "";
    for (i = 0; i < nodes.length; i++){
	ocamlcode +=nodes[i].textContent.replace(/        /g,"\t").replace(/  /g,"\n");
    }*/
    var code = "";
    var cells = IPython.notebook.get_cells();
    
    for (i=0; i< cells.length; i++){
	if (cells[i].cell_type === "code"){
	    code += cells[i].get_text()+"\n";
	}
	else
	{
	    code += "(* cell_type : "+cells[i].cell_type+" *)\n";
	    code += "(* " + cells[i].get_text() + " *)\n";
	}
    }
    var blob = new Blob([code], {type : "text/plain"});
    title = document.getElementsByClassName("text_cell_render border-box-sizing rendered_html")[0].textContent;
    saveAs(blob, title+".ml");
}

function downloadIPYNB () {
/*    var nodes = document.getElementsByClassName("CodeMirror cm-s-ipython");*/

    var ipynb = JSON.stringify(IPython.notebook.toJSON());
    var blob = new Blob([ipynb], {type : "text/plain"});
    title = document.getElementsByClassName("text_cell_render border-box-sizing rendered_html")[0].textContent;
    saveAs(blob, title+".ipynb");

}




function loadIPYNB (){
    var input=document.createElement('input');
    input.type="file";
    input.size="1";

    $(input).click();
    input.addEventListener("change", handleFiles, false);
    var fr = new FileReader();
    function handleFiles() {
	var files_read = input.files;    
	fr.readAsText(files_read[0]);
    };

    fr.onload = function(e) {

	var r = e.target.result;
	IPython.notebook.fromJSON(JSON.parse(r))
    };


}

// (function () {
//     var viewFullScreen = document.getElementById("slideshow");
//     if (viewFullScreen) {
//     }
// }) ;

var currentCell = 0;
var fullScreenDiv;

function press(evt) 
{
//    console.log(evt);

    //evt = window.event;
    var code = evt.which || evt.keyCode;
    var viewFullScreen = document.getElementById("slideshow");
    var cells = IPython.notebook.get_cells();
    switch(code) 
    {
    case 27 /*Esc*/:
	for (var i = 0; i < cells.length; i++){
	    cells[i].element.show();
	}
	document.removeEventListener("keydown",press);
	break;
    case 112 /*f1*/: if (currentCell > 0){ 
	cells[currentCell].element.hide(); 
	currentCell--;  
	IPython.notebook.select_prev();} break;
    case 113 /*f2*/: if (currentCell < cells.length - 1 ) {
	cells[currentCell].element.hide(); 
	currentCell++; 
	IPython.notebook.select_next();} break;
    };
    cells[currentCell].element.show();
    return true;
}


function deepClone (elt){
    var target = elt.cloneNode(true);
    deepCopy(elt,target);
    return target;
}

function deepCopy (elt,cloned){
    var children = elt.childNodes;
    var clonedChildren = cloned.childNodes;
    for (var i = 0; i < children.length; i++){

	if (children[i].nodeName == "CANVAS"){
	    var context = clonedChildren[i].getContext('2d');
	    clonedChildren[i].width = children[i].width;
	    clonedChildren[i].height = children[i].height;
	    context.drawImage(children[i], 0, 0);
	}
	deepCopy(children[i],clonedChildren[i]);
    }
}

function lol () {
    var cells = IPython.notebook.get_cells();
    for (var i = 0; i < cells.length; i++){
	cells[i].element.hide();
    }
    if (screenfull.enabled) {
     	var cellElt = cells[currentCell].element[0];
	document.addEventListener("keydown", press, false);
	fullScreenDiv.appendChild(deepClone(cellElt));
	cells[currentCell].element.show();
	var app= document.getElementById("notebook");
	screenfull.request(app);
    }
}

(function () {
    var viewFullScreen = document.getElementById("slideshow");
    fullScreenDiv = document.createElement("div");
    document.body.appendChild(fullScreenDiv);
    if (viewFullScreen) {
	viewFullScreen.addEventListener("click", lol, false);
    }
})();




function loadSlideshow () {
    var viewFullScreen = document.getElementById("slideshow");
//    viewFullScreen.click();
}
 

